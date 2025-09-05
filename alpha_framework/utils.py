from __future__ import annotations
from typing import Dict, Tuple

import numpy as np

from .alpha_framework_types import TypeId, OpSpec

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
