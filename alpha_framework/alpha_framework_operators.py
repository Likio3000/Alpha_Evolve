from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

# Imports from the types module
from .alpha_framework_types import TypeId, OpSpec, OP_REGISTRY, SAFE_MAX, EPS

###############################################################################
# 2 -- Primitive operators (and helpers)
###############################################################################

def _clean_num(x):
    """
    Convert NaN → 0,  +inf → +SAFE_MAX,  -inf → -SAFE_MAX,
    then hard-clip to ±SAFE_MAX.  Works on scalars & nd-arrays.
    """
    return np.clip(
        np.nan_to_num(x, nan=0.0, posinf=SAFE_MAX, neginf=-SAFE_MAX),
        -SAFE_MAX, SAFE_MAX
    )

def safe_op(fn):
    """
    Decorator: sanitise inputs **and** output of an element-wise numpy op.
      * No RuntimeWarnings leak out.
      * Result is always finite & within ±SAFE_MAX.
    """
    def wrapped(*args):
        clean_args = [_clean_num(a) for a in args]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            res = fn(*clean_args)
        return _clean_num(res)
    return wrapped

def register_op(name: str, *, in_types: Tuple[TypeId, ...], out: TypeId,
                is_cross_sectional_aggregator: bool = False,
                is_elementwise: bool = False):
    def _wrapper(fn: Callable):
        if name in OP_REGISTRY:
            raise KeyError(f"opcode '{name}' registered twice")
        OP_REGISTRY[name] = OpSpec(fn, in_types, out, is_cross_sectional_aggregator, is_elementwise)
        return fn
    return _wrapper


def _average_rank_ties_1d(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks [0..n-1] with stable sorting.

    Returns float ranks with ties assigned the mean of their index ranges.
    """
    n = x.size
    if n == 0:
        return x.astype(float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    xs = x[order]
    boundaries = np.empty(n + 1, dtype=bool)
    boundaries[0] = True
    if n > 1:
        boundaries[1:-1] = xs[1:] != xs[:-1]
    else:
        boundaries[1:-1] = False
    boundaries[-1] = True
    idx = np.flatnonzero(boundaries)
    for i in range(len(idx) - 1):
        start = idx[i]
        end = idx[i + 1]
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
    return ranks


@register_op("add", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _add(a, b): return a + b

@register_op("sub", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _sub(a, b): return a - b

@register_op("mul", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
@safe_op
def _mul(a, b): return a * b

@register_op("div", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _div(a, b):
    b_arr = np.asarray(b)
    a_arr_for_out = np.asarray(a) if not np.isscalar(a) else a
    out_shape_ref = a_arr_for_out if hasattr(a_arr_for_out, 'shape') else b_arr

    res = np.divide(a, b_arr,
                    out=np.zeros_like(out_shape_ref, dtype=float) if hasattr(out_shape_ref, 'shape') else 0.0,
                    where=np.abs(b_arr) > EPS)
    if np.isscalar(a) and np.isscalar(b):
        return res.item() if isinstance(res, np.ndarray) else res
    return res


@register_op("tanh", in_types=("scalar",), out="scalar", is_elementwise=True)
def _tanh(a): return np.tanh(a)

@register_op("sign", in_types=("scalar",), out="scalar", is_elementwise=True)
def _sign(a): return np.sign(a)

@register_op("neg", in_types=("scalar",), out="scalar", is_elementwise=True)
def _neg(a): return -a

@register_op("abs", in_types=("scalar",), out="scalar", is_elementwise=True)
def _abs(a): return np.abs(a)

@register_op("log", in_types=("scalar",), out="scalar", is_elementwise=True)
def _log(a):
    return np.log(np.maximum(np.asarray(a), EPS))

@register_op("sqrt", in_types=("scalar",), out="scalar", is_elementwise=True)
def _sqrt(a): return np.sqrt(np.maximum(np.asarray(a), 0))

@register_op("exp", in_types=("scalar",), out="scalar", is_elementwise=True)
@safe_op
def _exp(a):
    return np.exp(a)

@register_op("sin", in_types=("scalar",), out="scalar", is_elementwise=True)
@safe_op
def _sin(a):
    return np.sin(a)

@register_op("cos", in_types=("scalar",), out="scalar", is_elementwise=True)
@safe_op
def _cos(a):
    return np.cos(a)

@register_op("tan", in_types=("scalar",), out="scalar", is_elementwise=True)
@safe_op
def _tan(a):
    return np.tan(a)

@register_op("heaviside", in_types=("scalar",), out="scalar", is_elementwise=True)
def _heaviside(a):
    a_arr = np.asarray(a)
    return np.where(a_arr > 0, 1.0, 0.0)

@register_op("power", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _power(a, b):
    """
    Safe element-wise exponentiation.

    * Silences overflow / invalid / divide warnings from NumPy.
    * Cleans NaN/±inf → finite.
    * Hard-clips result to ±MAX_ABS so numbers never explode downstream.
    """
    MAX_ABS = 1e9 # This MAX_ABS is specific to _power, SAFE_MAX is for general cleaning
    a_arr = np.asarray(a, dtype=float)
    b_is_scalar = np.isscalar(b)
    b_val = b if b_is_scalar else np.asarray(b, dtype=float)

    # ------------------------------------------------------------------ #
    def _clip(x):
        return np.clip(x, -MAX_ABS, MAX_ABS)

    # ------------------------- scalar exponent ------------------------ #
    if b_is_scalar:
        if b_val == 2:
            return _clip(np.square(a_arr))
        if b_val == 1:
            return _clip(a_arr)
        if b_val == 0:
            return np.ones_like(a_arr, dtype=float)
        if b_val == 0.5:
            return _clip(np.sqrt(np.maximum(a_arr, 0.0)))
        if b_val == -1:
            return _clip(_div(1.0, a_arr))  # Uses _div from this module

        is_a_zero = np.all(np.abs(a_arr) < EPS) if isinstance(a_arr, np.ndarray) \
                    else abs(a_arr) < EPS
        if is_a_zero:
            if b_val > 0:
                return np.zeros_like(a_arr, dtype=float)
            if b_val == 0:
                return np.ones_like(a_arr, dtype=float)
            return np.copysign(np.full_like(a_arr, np.inf, dtype=float), a_arr)

        is_b_int = isinstance(b_val, (int, float)) and b_val == round(b_val)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            if (np.any(a_arr < 0) and not is_b_int):
                result = np.sign(a_arr) * np.power(np.abs(a_arr), b_val)
            else:
                result = np.power(a_arr, b_val)

    # ------------------------- array exponent ------------------------- #
    else:
        b_arr = b_val
        result = np.full_like(a_arr, np.nan, dtype=float)

        mask0 = np.abs(a_arr) < EPS
        result[mask0 & (b_arr > 0)]  = 0.0
        result[mask0 & (b_arr == 0)] = 1.0
        zneg = mask0 & (b_arr < 0)
        if np.any(zneg):
            result[zneg] = np.copysign(np.inf, a_arr[zneg])

        mask_safe = (~mask0) & ((a_arr > 0) | (np.isclose(b_arr, np.round(b_arr))))
        mask_signed = (~mask0) & (a_arr < 0) & (~np.isclose(b_arr, np.round(b_arr)))

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            if np.any(mask_safe):
                result[mask_safe] = np.power(a_arr[mask_safe], b_arr[mask_safe])
            if np.any(mask_signed):
                result[mask_signed] = np.sign(a_arr[mask_signed]) * \
                                      np.power(np.abs(a_arr[mask_signed]),
                                               b_arr[mask_signed])

    # ------------------------- clean-up + clamp ----------------------- #
    result = np.nan_to_num(result,
                           nan=0.0,
                           posinf=np.finfo(float).max/2,
                           neginf=-np.finfo(float).max/2)
    return _clip(result)


@register_op("min_val", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _min_val(a, b): return np.minimum(a, b)

@register_op("max_val", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _max_val(a, b): return np.maximum(a, b)


# Vector ops
@register_op("cs_mean", in_types=("vector",), out="scalar", is_cross_sectional_aggregator=True)
def _cs_mean(v): return float(np.mean(v)) if v.size > 0 else 0.0

@register_op("cs_std", in_types=("vector",), out="scalar", is_cross_sectional_aggregator=True)
def _cs_std(v): return float(np.std(v, ddof=0)) if v.size > 1 else 0.0

@register_op("cs_rank", in_types=("vector",), out="vector")
def _cs_rank(v):
    if v.size <= 1:
        return np.zeros_like(v)
    ranks = _average_rank_ties_1d(v.astype(float, copy=False))
    return (ranks / (len(v) - 1 + 1e-9)) * 2.0 - 1.0

@register_op("cs_demean", in_types=("vector",), out="vector")
def _cs_demean(v): return v - (_cs_mean(v) if v.size > 0 else 0.0)

@register_op("relation_rank", in_types=("vector", "vector"), out="vector")
def _relation_rank(v, groups):
    v_arr = np.asarray(v, dtype=float)
    grp_arr = np.asarray(groups, dtype=int)
    n = v_arr.size
    out = np.zeros_like(v_arr, dtype=float)
    if n <= 1:
        return out
    # Process per-group using a single stable sort to improve cache locality
    order = np.lexsort((v_arr, grp_arr))  # primary: grp, secondary: value
    gs = grp_arr[order]
    xs = v_arr[order]
    # group boundaries
    boundaries = np.empty(n + 1, dtype=bool)
    boundaries[0] = True
    if n > 1:
        boundaries[1:-1] = gs[1:] != gs[:-1]
    else:
        boundaries[1:-1] = False
    boundaries[-1] = True
    idx = np.flatnonzero(boundaries)
    tmp = np.empty(n, dtype=float)
    for i in range(len(idx) - 1):
        start = idx[i]
        end = idx[i + 1]
        if end - start <= 1:
            tmp[start:end] = 0.0
            continue
        ranks = _average_rank_ties_1d(xs[start:end])
        tmp[start:end] = (ranks / (float(end - start) - 1.0 + EPS)) * 2.0 - 1.0
    out[order] = tmp
    return out

@register_op("relation_demean", in_types=("vector", "vector"), out="vector")
def _relation_demean(v, groups):
    v_arr = np.asarray(v, dtype=float)
    grp_arr = np.asarray(groups, dtype=int)
    out = np.empty_like(v_arr, dtype=float)
    # Map group ids to 0..G-1 for compact accumulation
    uniq, inv = np.unique(grp_arr, return_inverse=True)
    g = inv
    G = uniq.size
    sums = np.zeros(G, dtype=float)
    counts = np.zeros(G, dtype=float)
    np.add.at(sums, g, v_arr)
    np.add.at(counts, g, 1.0)
    means = np.zeros(G, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    out = v_arr - means[g]
    return out

@register_op("vec_add_scalar", in_types=("vector", "scalar"), out="vector")
def _vec_add_scalar(v, s): return v + s

@register_op("vec_mul_scalar", in_types=("vector", "scalar"), out="vector")
@safe_op
def _vec_mul_scalar(v, s): return v * s

@register_op("vec_div_scalar", in_types=("vector", "scalar"), out="vector")
@safe_op  # safe_op will handle _clean_num for inputs
def _vec_div_scalar(v, s):
    """Divide vector ``v`` by scalar ``s`` while avoiding zero denominators."""
    safe_sign = np.sign(s) if abs(s) > EPS else 1.0
    denom = safe_sign * max(abs(s), 1e-3)
    return v / denom


# Matrix ops
@register_op("matmul_mv", in_types=("matrix", "vector"), out="vector")
def _matmul_mv(m, v): return m @ v

@register_op("matmul_mm", in_types=("matrix", "matrix"), out="matrix")
def _matmul_mm(m1, m2): return m1 @ m2

@register_op("transpose", in_types=("matrix",), out="matrix")
def _transpose(m): return m.T

@register_op("norm", in_types=("matrix",), out="scalar")
def _norm(m):
    return float(np.linalg.norm(m))

# Extraction ops
@register_op("get_feature_vector", in_types=("matrix", "scalar"), out="vector")
def _get_feature_vector(m, idx):
    safe_idx = int(np.clip(np.round(idx), 0, m.shape[0]-1))
    return m[safe_idx, :]

@register_op("get_stock_vector", in_types=("matrix", "scalar"), out="vector")
def _get_stock_vector(m, idx):
    safe_idx = int(np.clip(np.round(idx), 0, m.shape[1]-1))
    return m[:, safe_idx]

# Identity/Assignment
@register_op("assign_vector", in_types=("vector",), out="vector")
def _assign_vector(v): return np.array(v, copy=True)

@register_op("assign_scalar", in_types=("scalar",), out="scalar")
def _assign_scalar(s): return s

@register_op("assign_matrix", in_types=("matrix",), out="matrix")
def _assign_matrix(m): return np.array(m, copy=True)
