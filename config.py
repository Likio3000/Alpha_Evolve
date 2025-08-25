# config.py
"""
Dataclass-based “single source of truth” for every pipeline knob.

* DataConfig       – parameters that influence how OHLCV is loaded/aligned
* EvolutionConfig  – everything the evolutionary search needs
* BacktestConfig   – everything the cross-sectional back-tester needs
"""

from dataclasses import dataclass, field
from typing import Dict

# Default mapping from token symbol to sector ID used across the project.  The
# mapping groups major crypto assets into rough "sectors" so relation-based
# operations can reason about them.  A value of ``-1`` denotes an unknown
# sector.
DEFAULT_CRYPTO_SECTOR_MAPPING: Dict[str, int] = {
    "BTC": 0,  # Core
    "ETH": 1,
    "SOL": 1,  # Ecosystem
    # Altcoins
    "ADA": 2,
    "AVA": 2,
    "SUI": 2,
    "APT": 2,
    "INJ": 2,
    "RNDR": 2,
    "ARB": 2,
    "LINK": 2,
    # Memes
    "BONK": 3,
    "DOGE": 3,
    "PEPE": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
#  shared data-handling knobs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    data_dir: str = "./data"
    max_lookback_data_option: str = "common_1200"
    min_common_points: int = 1200
    eval_lag: int = 1
    # Mapping from token symbols to sector IDs used for relation-aware
    # operations.  It defaults to ``DEFAULT_CRYPTO_SECTOR_MAPPING`` defined
    # above but can be overridden per configuration instance.
    sector_mapping: Dict[str, int] = field(
        default_factory=lambda: DEFAULT_CRYPTO_SECTOR_MAPPING.copy()
    )


# ─────────────────────────────────────────────────────────────────────────────
#  evolution search
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvolutionConfig(DataConfig):
    generations: int = 10
    seed: int = 42
    pop_size: int = 100
    tournament_k: int = 10
    elite_keep: int = 1
    hof_size: int = 20
    # Add top-K candidates per generation to the Hall of Fame (subject to
    # correlation and duplicate filters). Increases diversity of saved alphas.
    hof_per_gen: int = 3
    quiet: bool = False

    # breeding / variation
    p_mut: float = 0.9
    p_cross: float = 0.0
    fresh_rate: float = 0.12

    # complexity / similarity guards
    # operation limits (Section 4.1 of the paper)
    max_ops: int = 87
    max_setup_ops: int = 21
    max_predict_ops: int = 21
    max_update_ops: int = 45
    max_scalar_operands: int = 10
    max_vector_operands: int = 16
    max_matrix_operands: int = 4
    parsimony_penalty: float = 0.002
    corr_penalty_w: float = 0.35
    corr_cutoff: float = 0.15
    sharpe_proxy_w: float = 0.0
    ic_std_penalty_w: float = 0.10
    turnover_penalty_w: float = 0.05
    use_train_val_splits: bool = True
    train_points: int = 840
    val_points: int = 360
    keep_dupes_in_hof: bool = False

    # ops variance controls
    # Introduce deterministic jitter to the parsimony penalty per program
    # to avoid collapsing to extreme sizes. 0.0 disables the feature.
    parsimony_jitter_pct: float = 0.0

    # evaluation specifics
    xs_flat_guard: float = 5e-2
    t_flat_guard: float = 5e-2
    early_abort_bars: int = 20
    early_abort_xs: float = 5e-2
    early_abort_t: float = 5e-2
    flat_bar_threshold: float = 0.25
    scale: str = "madz"
    # Optional preprocessing tweaks
    sector_neutralize: bool = True       # Demean positions by sector before IC
    winsor_p: float = 0.02               # Tail prob for 'winsor' scale

    # evaluation cache
    eval_cache_size: int = 128

    # multiprocessing
    workers: int = 1

    # random program generation variance
    # Jitter the (setup, predict, update) op split when seeding fresh programs.
    # 0.0 keeps fixed ~[15%, 70%, 15%]; higher values add randomness.
    ops_split_jitter: float = 0.0

    # ramp scheduling for annealed penalties (corr, ic_std, turnover, sharpe_proxy)
    ramp_fraction: float = 1.0/3.0  # portion of total gens to reach full weight
    ramp_min_gens: int = 5          # minimum generations to ramp over

    # selection criterion for breeding/elites while logging can still show ramped
    # Options: 'ramped' (default fitness), 'fixed' (fitness_static), 'ic' (mean_ic)
    selection_metric: str = "ramped"


# ─────────────────────────────────────────────────────────────────────────────
#  cross-sectional back-test
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig(DataConfig):
    top_to_backtest: int = 10
    fee: float = 1.0                      # round-trip commission (bps)
    hold: int = 1                         # holding period (bars)
    scale: str = "zscore"
    long_short_n: int = 0                 # 0 → use all symbols
    # For 4 hour bars on 24/7 crypto data use 365 days × 6 bars/day
    annualization_factor: float = 365 * 6
    seed: int = 42
    # Optional intrabar per-asset stop-loss. If > 0, a long exits at
    # -stop_loss_pct and a short exits at +stop_loss_pct during the next bar
    # when breached (requires lag=1). A one-way fee is charged for the stop exit.
    stop_loss_pct: float = 0.0
    # Risk controls (optional; disabled by default)
    # Sector-neutralize daily target positions before final normalization.
    sector_neutralize_positions: bool = False
    # Volatility targeting on portfolio returns (daily). If <= 0, disabled.
    volatility_target: float = 0.0        # target daily vol (e.g., 0.01 for 1%)
    volatility_lookback: int = 30         # bars to estimate realized vol
    max_leverage: float = 2.0             # cap on exposure multiplier
    min_leverage: float = 0.25            # floor on exposure multiplier
    # Drawdown limiter: reduce exposure when DD exceeds threshold.
    dd_limit: float = 0.0                 # e.g., 0.15 for 15%; if <=0 disabled
    dd_reduction: float = 0.5             # multiply exposure when beyond dd_limit


# keep old import path alive
EvoConfig = EvolutionConfig
