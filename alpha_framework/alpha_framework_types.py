from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Literal

# ---------------------------------------------------------------------------
# 1 ‑‑ Type system (and core constants)
# ---------------------------------------------------------------------------

TypeId = Literal["scalar", "vector", "matrix"]

CROSS_SECTIONAL_FEATURE_VECTOR_NAMES = [
    "opens_t", "highs_t", "lows_t", "closes_t", "ranges_t",
    "ma5_t", "ma10_t", "ma20_t", "ma30_t",
    "vol5_t", "vol10_t", "vol20_t", "vol30_t",
    "sector_id_vector",
]
SCALAR_FEATURE_NAMES = ["const_1", "const_neg_1"]

FINAL_PREDICTION_VECTOR_NAME = "s1_predictions_vector"

SAFE_MAX = 1e6  # Moved here as it's a general constant for safe operations

@dataclass
class OpSpec:
    func: Callable
    in_types: Tuple[TypeId, ...]
    out_type: TypeId
    is_cross_sectional_aggregator: bool = False
    is_elementwise: bool = False

OP_REGISTRY: Dict[str, OpSpec] = {}