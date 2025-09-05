from __future__ import annotations
from typing import Dict, Tuple, Iterable, List, Optional

import numpy as np

from .alpha_framework_types import TypeId, OpSpec
from .alpha_framework_types import CROSS_SECTIONAL_FEATURE_VECTOR_NAMES

# Optional weighting to favour relation-aware and cross-sectional ops
RELATION_OPS_WEIGHT = 3.0
CS_OPS_WEIGHT = 1.5
DEFAULT_OP_WEIGHT = 1.0


def op_weight(op_name: str, *, is_predict: bool) -> float:
    """Heuristic weight for picking ops during generation/mutation."""
    w = DEFAULT_OP_WEIGHT
    if op_name.startswith("relation_"):
        w *= RELATION_OPS_WEIGHT
    if op_name.startswith("cs_"):
        w *= CS_OPS_WEIGHT
    if is_predict and (op_name.startswith("relation_") or op_name.startswith("cs_")):
        w *= 1.2
    return w


def effective_out_type(
    spec: OpSpec, available_types: Dict[str, TypeId], inputs: Tuple[str, ...]
) -> TypeId:
    """
    Infer the actual output type for an op given current input variable types.

    Elementwise ops that nominally output scalars are promoted to vectors when
    any input that the spec marks as "scalar" is provided a vector at runtime.
    """
    out_t: TypeId = spec.out_type
    if spec.is_elementwise and spec.out_type == "scalar":
        for i, name in enumerate(inputs):
            # Only consider promotion when a scalar slot receives a vector
            if spec.in_types[i] == "scalar" and available_types.get(name) == "vector":
                out_t = "vector"
                break
    return out_t


def broadcast_to_vector(x: float | np.ndarray, n: int) -> np.ndarray:
    """Broadcast a scalar or 1-element array to a length-n vector."""
    if np.isscalar(x):
        return np.full(n, float(x))
    arr = np.asarray(x)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(n, float(arr.item()))
    if arr.ndim != 1:
        raise TypeError(f"Expected scalar or 1D array to broadcast, got shape {arr.shape}")
    # If length already matches, return as is
    if arr.shape[0] == n:
        return arr
    # Fallback: copy/min-pad or trim to match length
    out = np.zeros(n, dtype=arr.dtype)
    m = min(n, arr.shape[0])
    out[:m] = arr[:m]
    return out


def ensure_vector_shape(
    x: float | np.ndarray,
    n: int,
    *,
    allow_mismatch_for_aggregator: bool = False,
) -> np.ndarray:
    """
    Ensure `x` is a length-n 1D vector. Scalars are broadcast; vectors are
    resized if needed unless `allow_mismatch_for_aggregator` is True.
    """
    if np.isscalar(x):
        return np.full(n, float(x))
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise TypeError(f"Expected 1D vector, got array with shape {arr.shape}")
    if arr.shape[0] == n or allow_mismatch_for_aggregator:
        return arr
    # Resize conservatively when shapes drift
    out = np.zeros(n, dtype=arr.dtype)
    m = min(n, arr.shape[0])
    out[:m] = arr[:m]
    return out


# ------------------------------ helper: candidates ------------------------------

def select_var_candidates(
    vars_map: Dict[str, TypeId],
    required_type: TypeId,
    *,
    allow_elementwise_scalar_promotion: bool = False,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Build a weighted list of candidate variable names matching `required_type`.

    - For scalars, prefer non-constant scalars over constants by repeating them.
    - If no scalars exist and promotion is allowed, use vectors as fallback.
    - `exclude` allows leaving out current inputs when re-wiring.
    """
    exclude_set = set(exclude or ())

    if required_type == "scalar":
        # Defer import to avoid cycle
        from .alpha_framework_types import SCALAR_FEATURE_NAMES

        consts: List[str] = []
        others: List[str] = []
        for name, t in vars_map.items():
            if name in exclude_set:
                continue
            if t == "scalar":
                if name in SCALAR_FEATURE_NAMES:
                    consts.append(name)
                else:
                    others.append(name)
        weighted: List[str] = []
        if others:
            weighted.extend(others * 3)
            weighted.extend(consts)
        elif consts:
            weighted.extend(consts)

        if not weighted and allow_elementwise_scalar_promotion:
            weighted.extend([n for n, t in vars_map.items() if t == "vector" and n not in exclude_set])
        return weighted

    # vector or matrix
    return [n for n, t in vars_map.items() if t == required_type and n not in exclude_set]


def pick_vector_fallback(
    vars_before: Dict[str, TypeId],
    feature_vars: Dict[str, TypeId],
    *,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Choose a reasonable vector source with graceful fallbacks.
    Order: available vars → feature vars → default list → 'opens_t'.
    """
    rng = rng or np.random.default_rng()
    options = [n for n, t in vars_before.items() if t == "vector"]
    if options:
        return str(rng.choice(options))

    feat_opts = [n for n, t in feature_vars.items() if t == "vector"]
    if feat_opts:
        return str(rng.choice(feat_opts))

    if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
        return CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0]
    return "opens_t"


def temp_name(out_type: TypeId, *, rng: Optional[np.random.Generator] = None, prefix: str = "m") -> str:
    """Generate a temporary variable name that encodes type and randomness."""
    rng = rng or np.random.default_rng()
    tag = {"scalar": "s", "vector": "v", "matrix": "m"}[out_type]
    return f"{prefix}{int(rng.integers(10_000, 99_999))}_{tag}"
