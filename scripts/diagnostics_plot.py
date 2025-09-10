#!/usr/bin/env python
"""
diagnostics_plot.py – Generate evolution graphs from diagnostics.json.

Usage:
  uv run scripts/diagnostics_plot.py /path/to/run_dir

If no path is given, reads the path from pipeline_runs_cs/LATEST.
Outputs PNG files next to diagnostics.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

# Force a non-interactive backend for headless environments
try:
    import matplotlib
    matplotlib.use("Agg")  # Safe in both headless and GUI envs
except Exception:
    # If matplotlib is not installed, the import below will handle it
    pass


def _resolve_run_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.is_dir():
            return p
        raise SystemExit(f"Run dir does not exist: {p}")
    latest = Path("pipeline_runs_cs") / "LATEST"
    if latest.exists():
        return Path(latest.read_text().strip())
    raise SystemExit("No run dir argument and LATEST not found.")


def _load_diags(run_dir: Path):
    diag_path = run_dir / "diagnostics.json"
    if not diag_path.exists():
        raise SystemExit(f"diagnostics.json not found in {run_dir}")
    with open(diag_path) as fh:
        return json.load(fh)


def _to_series(diags: list[dict], key_path: list[str], default=None):
    out = []
    for d in diags:
        x = d
        for k in key_path:
            x = x.get(k, {}) if isinstance(x, dict) else {}
        if isinstance(x, dict):
            out.append(default)
        else:
            out.append(x if x is not None else default)
    return out


def generate_plots(run_dir: Path) -> Path:
    """Generate diagnostic plots for a specific run directory.

    Returns the directory where plots were written.
    """
    diags = _load_diags(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots. Install matplotlib to enable.")
        return plots_dir

    # Apply a dark theme to match the dashboard UI
    try:
        plt.rcParams.update({
            "figure.facecolor": "#0f1317",
            "axes.facecolor": "#12161a",
            "savefig.facecolor": "#0f1317",
            "savefig.edgecolor": "#0f1317",
            "axes.edgecolor": "#2a3138",
            "axes.labelcolor": "#e6e8eb",
            "text.color": "#e6e8eb",
            "xtick.color": "#a7adb3",
            "ytick.color": "#a7adb3",
            "grid.color": "#1f242a",
        })
    except Exception:
        pass

    gens = np.array([int(d.get("generation", i + 1)) for i, d in enumerate(diags)], dtype=int)

    def _safe_series(vals):
        arr = np.array(vals, dtype=float)
        # Replace None with nan
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    def _safe_plot(ax, x, y, *args, **kwargs):
        x_arr = np.asarray(x)
        y_arr = _safe_series(y)
        n = min(len(x_arr), len(y_arr))
        if n == 0:
            return False
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
        if not np.isfinite(y_arr).any():
            return False
        ax.plot(x_arr, y_arr, *args, **kwargs)
        return True

    # Fitness quantiles
    q_best = [d.get("pop_quantiles", {}).get("best") for d in diags]
    q_p95 = [d.get("pop_quantiles", {}).get("p95") for d in diags]
    q_med = [d.get("pop_quantiles", {}).get("median") for d in diags]
    q_p25 = [d.get("pop_quantiles", {}).get("p25") for d in diags]

    fig, ax = plt.subplots(figsize=(8, 4))
    _safe_plot(ax, gens, q_best, label="Best", lw=2)
    _safe_plot(ax, gens, q_p95, label="P95", alpha=0.7)
    # Rolling-window smoothing for median
    med_arr = _safe_series(q_med)
    win = max(3, min(9, len(med_arr)//10*2+1))  # odd window ~10% of run, min 3, max 9
    if np.isfinite(med_arr).any() and win > 1:
        valid = np.isfinite(med_arr)
        smooth = _moving_avg_nan(med_arr, win)
        _safe_plot(ax, gens, smooth, label=f"Median (smoothed w={win})", lw=2)
    else:
        _safe_plot(ax, gens, q_med, label="Median", lw=2)
    _safe_plot(ax, gens, q_p25, label="P25", alpha=0.7)
    ax.set_title("Fitness progress")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    (plots_dir / "fitness_quantiles.png").write_bytes(fig_to_png(fig))
    plt.close(fig)

    # Best program components
    best_fit = [d.get("best", {}).get("fitness") for d in diags]
    best_ic = [d.get("best", {}).get("mean_ic") for d in diags]
    fig, ax = plt.subplots(figsize=(8, 4))
    _safe_plot(ax, gens, best_fit, label="Best fitness", lw=2)
    _safe_plot(ax, gens, best_ic, label="Best IC", lw=2)
    ax.set_title("Best-of-generation metrics")
    ax.set_xlabel("Generation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    (plots_dir / "best_metrics.png").write_bytes(fig_to_png(fig))
    plt.close(fig)

    # Early aborts / guards
    e_xs = [d.get("eval_stats", {}).get("early_abort_xs") for d in diags]
    e_t = [d.get("eval_stats", {}).get("early_abort_t") for d in diags]
    e_fb = [d.get("eval_stats", {}).get("early_abort_flatbar") for d in diags]
    fig, ax = plt.subplots(figsize=(8, 4))
    _safe_plot(ax, gens, e_xs, label="early_abort_xs")
    _safe_plot(ax, gens, e_t, label="early_abort_t")
    _safe_plot(ax, gens, e_fb, label="early_abort_flatbar")
    ax.set_title("Early aborts / flatness guards")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count (sampled)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    (plots_dir / "early_aborts.png").write_bytes(fig_to_png(fig))
    plt.close(fig)

    # Ramp weights
    r_corr = [d.get("ramp", {}).get("corr_w") for d in diags]
    r_icstd = [d.get("ramp", {}).get("ic_std_w") for d in diags]
    r_turn = [d.get("ramp", {}).get("turnover_w") for d in diags]
    r_sh = [d.get("ramp", {}).get("sharpe_w") for d in diags]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    _safe_plot(ax, gens, r_corr, label="corr_w")
    _safe_plot(ax, gens, r_icstd, label="ic_std_w")
    _safe_plot(ax, gens, r_turn, label="turnover_w")
    _safe_plot(ax, gens, r_sh, label="sharpe_w")
    ax.set_title("Penalty weights (ramp)")
    ax.set_xlabel("Generation")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    (plots_dir / "ramp_weights.png").write_bytes(fig_to_png(fig))
    plt.close(fig)

    # Scatter: fitness vs ops for top-K across gens (colored by generation)
    ops = []
    fit = []
    gen_col = []
    for g, d in zip(gens, diags):
        for tk in d.get("topK", []) or []:
            ops.append(tk.get("ops"))
            fit.append(tk.get("fitness"))
            gen_col.append(g)
    if ops and fit:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sc = ax.scatter(ops, fit, c=gen_col, cmap="viridis", alpha=0.6, s=18, edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Generation")
        ax.set_title("Top-K programs: Fitness vs Ops")
        ax.set_xlabel("Ops")
        ax.set_ylabel("Fitness")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        (plots_dir / "topk_fitness_vs_ops.png").write_bytes(fig_to_png(fig))
        plt.close(fig)

    print(f"Saved plots → {plots_dir}")
    return plots_dir


def main() -> None:
    run_dir = _resolve_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    generate_plots(run_dir)


def fig_to_png(fig) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    return buf.getvalue()


def _moving_avg_nan(a: np.ndarray, window: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if window <= 1:
        return a
    w = np.ones(window, dtype=float)
    vals = np.convolve(np.nan_to_num(a, nan=0.0), w, mode="same")
    cnts = np.convolve(np.isfinite(a).astype(float), w, mode="same")
    out = np.full_like(vals, np.nan, dtype=float)
    nz = cnts > 0
    out[nz] = vals[nz] / cnts[nz]
    return out


if __name__ == "__main__":
    main()
