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
    # How numeric feature vectors are scaled cross-sectionally when extracted
    feature_scale_method: str = "zscore"


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
    keep_dupes_in_hof: bool = False

    # evaluation specifics
    xs_flat_guard: float = 5e-2
    t_flat_guard: float = 5e-2
    early_abort_bars: int = 20
    early_abort_xs: float = 5e-2
    early_abort_t: float = 5e-2
    flat_bar_threshold: float = 0.25
    scale: str = "zscore"

    # evaluation cache
    eval_cache_size: int = 128

    # multiprocessing
    workers: int = 1


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


# keep old import path alive
EvoConfig = EvolutionConfig
