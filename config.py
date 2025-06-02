# config.py
"""
Dataclass-based “single source of truth” for every pipeline knob.

* DataConfig       – parameters that influence how OHLCV is loaded/aligned
* EvolutionConfig  – everything the evolutionary search needs
* BacktestConfig   – everything the cross-sectional back-tester needs
"""

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
#  shared data-handling knobs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    data_dir: str = "./data"
    max_lookback_data_option: str = "common_1200"
    min_common_points: int = 1200
    eval_lag: int = 1


# ─────────────────────────────────────────────────────────────────────────────
#  evolution search
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvolutionConfig(DataConfig):
    generations: int = 10
    seed: int = 42
    pop_size: int = 96
    tournament_k: int = 3
    elite_keep: int = 4
    hof_size: int = 20
    quiet: bool = False

    # breeding / variation
    p_mut: float = 0.55
    p_cross: float = 0.6
    fresh_rate: float = 0.12

    # complexity / similarity guards
    max_ops: int = 24
    parsimony_penalty: float = 0.02
    corr_penalty_w: float = 0.35
    corr_cutoff: float = 0.15
    keep_dupes_in_hof: bool = False

    # evaluation specifics
    xs_flat_guard: float = 5e-2
    t_flat_guard: float = 5e-2
    early_abort_bars: int = 20
    early_abort_xs: float = 5e-2
    early_abort_t: float = 5e-2
    scale: str = "zscore"


# ─────────────────────────────────────────────────────────────────────────────
#  cross-sectional back-test
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig(DataConfig):
    top_to_backtest: int = 10
    fee: float = 1.0                      # round-trip commission (bps)
    hold: int = 1                         # holding period (bars)
    scale: str = "zscore"
    annualization_factor: float = 252 * 6
    seed: int = 42


# keep old import path alive
EvoConfig = EvolutionConfig
