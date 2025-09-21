from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Literal

# ---------------------------------------------------------------------------
# 1 ‑‑ Type system (and core constants)
# ---------------------------------------------------------------------------

TypeId = Literal["scalar", "vector", "matrix"]

CROSS_SECTIONAL_FEATURE_VECTOR_NAMES = [
    "opens_t", "highs_t", "lows_t", "closes_t", "ranges_t",
    "ma5_t", "ma10_t", "ma20_t", "ma30_t", "ma60_t", "ma90_t",
    "vol5_t", "vol10_t", "vol20_t", "vol30_t", "vol60_t", "vol90_t",
    "vol_spread_5_20_t", "vol_spread_10_30_t", "vol_spread_20_60_t", "vol_spread_30_90_t",
    "vol_ratio_5_20_t", "vol_ratio_20_60_t",
    "trend_5_20_t", "intraday_ret_t",
    "ret1d_t", "range_rel_t", "flow_proxy_t", "whale_move_proxy_t",
    "onchain_activity_proxy_t", "onchain_velocity_proxy_t", "onchain_whale_proxy_t",
    "market_rel_close_t", "market_rel_ret1d_t", "market_zclose_t",
    "btc_ratio_proxy_t", "regime_volatility_t", "regime_momentum_t",
    "cross_btc_momentum_t", "sector_momentum_diff_t", "market_dispersion_t",
    "sector_id_vector",
]
SCALAR_FEATURE_NAMES = ["const_1", "const_neg_1"]

FINAL_PREDICTION_VECTOR_NAME = "s1_predictions_vector"

SAFE_MAX = 1e6  # Moved here as it's a general constant for safe operations
EPS = 1e-9      # Unified small epsilon for numeric guards

@dataclass
class OpSpec:
    func: Callable
    in_types: Tuple[TypeId, ...]
    out_type: TypeId
    is_cross_sectional_aggregator: bool = False
    is_elementwise: bool = False

OP_REGISTRY: Dict[str, OpSpec] = {}
