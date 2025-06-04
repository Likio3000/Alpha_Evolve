from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np

from .alpha_framework_types import TypeId, OP_REGISTRY, FINAL_PREDICTION_VECTOR_NAME
from .alpha_framework_op import Op

if TYPE_CHECKING:
    from .alpha_framework_program import (
        AlphaProgram,
    )  # For type hint of cls and return type


def generate_random_program_logic(
    cls: type,  # Actually AlphaProgram class
    feature_vars: Dict[str, TypeId],
    state_vars: Dict[str, TypeId],
    max_total_ops: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> AlphaProgram:
    """
    Build a random but type-correct AlphaProgram.
    This is the core logic for AlphaProgram.random_program.
    """
    rng = rng or np.random.default_rng()
    prog = cls()  # type: ignore # cls() will be an AlphaProgram instance

    # — split total op budget into three blocks —
    n_predict_ops = max(1, int(max_total_ops * 0.70))
    n_setup_ops = int(max_total_ops * 0.15)
    n_update_ops = max_total_ops - n_predict_ops - n_setup_ops

    tmp_idx = 0

    def _new_tmp(t: TypeId) -> str:
        nonlocal tmp_idx
        tmp_idx += 1
        return f"{'s' if t == 'scalar' else 'v' if t == 'vector' else 'm'}{tmp_idx}"

    # ────────────────── helper to append ops ──────────────────
    def _add_ops(
        block: List[Op], in_vars: Dict[str, TypeId], how_many: int, is_predict: bool
    ) -> None:
        current = in_vars.copy()

        for k in range(how_many):
            # 1. gather candidates whose input types we can satisfy
            candidates = []
            for op_name, spec in OP_REGISTRY.items():
                if op_name == "assign_vector" and is_predict and k != how_many - 1:
                    continue  # reserve assign_vector for emergency only

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
                    out_t = spec.out_type
                    if spec.is_elementwise and out_t == "scalar":
                        if any(current[i] == "vector" for i in ins):
                            out_t = "vector"
                    if out_t == "vector":
                        chosen_name, chosen_spec, chosen_ins = op_name, spec, ins
                        break

                if chosen_name is None:
                    # <-- this is the NEW behaviour (May-2025 change in original file)
                    raise RuntimeError(
                        "random_program(): could not find a vector-typed "
                        "operation for the final predict slot"
                    )
            else:
                idx = rng.integers(len(candidates))  # ← pick index
                chosen_name, chosen_spec, pools = candidates[idx]
                chosen_ins = tuple(rng.choice(p) for p in pools)
                out_t = chosen_spec.out_type
                if chosen_spec.is_elementwise and out_t == "scalar":
                    if any(current[i] == "vector" for i in chosen_ins):
                        out_t = "vector"

            # 3. emit op
            out_name = (
                FINAL_PREDICTION_VECTOR_NAME if last_predict_slot else _new_tmp(out_t)
            )  # type: ignore
            block.append(Op(out_name, chosen_name, chosen_ins))  # type: ignore
            current[out_name] = out_t  # type: ignore

    # ────────────────── actually build the three blocks ──────────────────
    _add_ops(prog.setup, {**feature_vars, **state_vars}, n_setup_ops, False)
    # Need AlphaProgram._trace_vars_for_block here.
    # This indicates that _trace_vars_for_block might need to be a static helper too, or passed.
    # For now, let's assume prog has _trace_vars_for_block method.
    after_setup = prog._trace_vars_for_block(prog.setup, {**feature_vars, **state_vars})

    _add_ops(prog.predict_ops, after_setup, n_predict_ops, True)
    after_predict = prog._trace_vars_for_block(prog.predict_ops, after_setup)

    _add_ops(prog.update_ops, after_predict, n_update_ops, False)
    return prog
