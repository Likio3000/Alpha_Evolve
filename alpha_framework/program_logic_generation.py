from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np

from .alpha_framework_op import Op
from .alpha_framework_types import TypeId, OP_REGISTRY, FINAL_PREDICTION_VECTOR_NAME
from .utils import effective_out_type, op_weight, EvolutionParams

# Probability that a newly added op must output a vector.  Can be
# overridden by callers (e.g. `evolve_alphas`) before program creation.
VECTOR_OPS_BIAS = 0.0

# Weighting logic moved to utils.op_weight

# Per-stage limits from the paper
MAX_SETUP_OPS = 21
MAX_PREDICT_OPS = 21
MAX_UPDATE_OPS = 45

if TYPE_CHECKING:
    from .alpha_framework_program import AlphaProgram # For type hint of cls and return type


def generate_random_program_logic(
    cls: type, # Actually AlphaProgram class
    feature_vars: Dict[str, TypeId],
    state_vars: Dict[str, TypeId],
    max_total_ops: int = 32,
    rng: Optional[np.random.Generator] = None,
    max_setup_ops: int = MAX_SETUP_OPS,
    max_predict_ops: int = MAX_PREDICT_OPS,
    max_update_ops: int = MAX_UPDATE_OPS,
    ops_split_jitter: float = 0.0,
    params: Optional[EvolutionParams] = None,
) -> AlphaProgram:
    """
    Build a random but type-correct AlphaProgram.
    This is the core logic for AlphaProgram.random_program.
    """
    rng = rng or np.random.default_rng()
    local_vector_bias = VECTOR_OPS_BIAS if params is None else params.vector_ops_bias
    prog = cls() # type: ignore # cls() will be an AlphaProgram instance

    # — split total op budget into three blocks, respecting per-stage limits —
    # Base split ~ [setup 15%, predict 70%, update 15%] with optional jitter for variance
    base = np.array(params.ops_split_base if params is not None else [0.15, 0.70, 0.15], dtype=float)
    # Prefer explicit argument if provided, otherwise pull from params
    j_in = ops_split_jitter if ops_split_jitter is not None else 0.0
    j = float(max(0.0, min(1.0, j_in if j_in > 0 else (params.ops_split_jitter if params is not None else 0.0))))
    if j > 1e-12:
        noise = rng.normal(0.0, 0.3 * j, size=3)  # std scaled so j in [0,1] is reasonable
        props = base + noise
        props = np.clip(props, 0.01, None)
        props = props / props.sum()
    else:
        props = base

    # Initial allocation
    target = np.maximum([0, 1, 0], np.round(props * max_total_ops).astype(int))
    # Ensure at least 1 predict op
    if target[1] < 1:
        target[1] = 1
    total = int(target.sum())
    # Adjust to exactly max_total_ops
    while total < max_total_ops:
        # add to the block with largest remaining proportion signal
        k = int(np.argmax(props))
        target[k] += 1
        total += 1
    while total > max_total_ops:
        # remove from the block with smallest proportion but keep predict >=1
        k = int(np.argmin(props))
        if k == 1 and target[1] <= 1:
            k = 0 if target[0] > 0 else 2
        if target[k] > 0:
            target[k] -= 1
            total -= 1

    n_setup_ops, n_predict_ops, n_update_ops = int(target[0]), int(target[1]), int(target[2])

    n_setup_ops = min(n_setup_ops, max_setup_ops)
    n_predict_ops = min(n_predict_ops, max_predict_ops)
    n_update_ops = min(n_update_ops, max_update_ops)

    tmp_idx = 0
    def _new_tmp(t: TypeId) -> str:
        nonlocal tmp_idx
        tmp_idx += 1
        return f"{'s' if t=='scalar' else 'v' if t=='vector' else 'm'}{tmp_idx}"

    # ────────────────── helper to append ops ──────────────────
    def _add_ops(block: List[Op],
                 in_vars: Dict[str, TypeId],
                 how_many: int,
                 is_predict: bool) -> None:

        current = in_vars.copy()

        for k in range(how_many):
            # 1. gather candidates whose input types we can satisfy
            candidates = []
            for op_name, spec in OP_REGISTRY.items():
                if op_name == "assign_vector" and is_predict and k != how_many - 1:
                    continue          # reserve assign_vector for emergency only

                # ─── bias towards ops that output vectors ───
                if rng.random() < local_vector_bias and spec.out_type != "vector":
                    continue

                inputs_for_spec: List[List[str]] = []
                ok = True
                for need_t in spec.in_types:
                    pool = [v for v, t in current.items() if t == need_t]

                    # allow scalar-slot promotion of a vector for element-wise ops
                    if not pool and need_t == "scalar" and spec.is_elementwise:
                        pool = [v for v, t in current.items() if t == "vector"]

                    if not pool:
                        ok = False
                        break
                    inputs_for_spec.append(pool)

                if ok:
                    candidates.append((op_name, spec, inputs_for_spec))

            # no viable op → give up for this position
            if not candidates:
                break

            last_predict_slot = is_predict and k == how_many - 1
            chosen_name = chosen_spec = chosen_ins = out_t = None

            # ─── we treat the final predict op specially ───
            if last_predict_slot:
                rng.shuffle(candidates)
                for op_name, spec, pools in candidates:
                    ins = tuple(rng.choice(p) for p in pools)
                    out_t = effective_out_type(spec, current, ins)
                    if out_t == "vector":
                        chosen_name, chosen_spec, chosen_ins = op_name, spec, ins
                        break

                if chosen_name is None:
                    # Emergency fallback: take any available vector input
                    fallback = next((vn for vn, vt in current.items() if vt == "vector"), None)
                    if fallback is None:
                        fallback = "opens_t"
                    block.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (fallback,)))
                    return prog
            else:
                # Weighted pick favouring relation_* and cs_* ops
                if candidates:
                    weights = np.array([op_weight(n, is_predict=is_predict, params=params) for n, _, _ in candidates], dtype=float)
                    if np.all(weights <= 0) or np.isnan(weights).any():
                        idx = rng.integers(len(candidates))
                    else:
                        weights = weights / weights.sum()
                        idx = int(rng.choice(len(candidates), p=weights))
                else:
                    idx = 0
                chosen_name, chosen_spec, pools = candidates[idx]
                chosen_ins = tuple(rng.choice(p) for p in pools)
                out_t = effective_out_type(chosen_spec, current, chosen_ins)

            # 3. emit op
            out_name = (FINAL_PREDICTION_VECTOR_NAME
                        if last_predict_slot
                        else _new_tmp(out_t)) # type: ignore
            block.append(Op(out_name, chosen_name, chosen_ins)) # type: ignore
            current[out_name] = out_t # type: ignore

    # ────────────────── actually build the three blocks ──────────────────
    _add_ops(prog.setup,   {**feature_vars, **state_vars},           n_setup_ops,   False)
    # Need AlphaProgram._trace_vars_for_block here.
    # This indicates that _trace_vars_for_block might need to be a static helper too, or passed.
    # For now, let's assume prog has _trace_vars_for_block method.
    after_setup = prog._trace_vars_for_block(prog.setup,
                                             {**feature_vars, **state_vars})

    _add_ops(prog.predict_ops, after_setup,                          n_predict_ops, True)
    after_predict = prog._trace_vars_for_block(prog.predict_ops,
                                               after_setup)

    _add_ops(prog.update_ops, after_predict,                         n_update_ops,  False)
    return prog
