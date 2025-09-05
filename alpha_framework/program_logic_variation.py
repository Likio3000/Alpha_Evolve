from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional
import numpy as np
import copy # For crossover, specifically deepcopy of other's predict_ops

from .alpha_framework_op import Op
from .program_logic_generation import (
    MAX_SETUP_OPS,
    MAX_PREDICT_OPS,
    MAX_UPDATE_OPS,
)
from .alpha_framework_types import (
    TypeId,
    OP_REGISTRY,
    FINAL_PREDICTION_VECTOR_NAME,
    SCALAR_FEATURE_NAMES,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
)
from .utils import (
    effective_out_type,
    op_weight,
    select_var_candidates,
    pick_vector_fallback,
    temp_name,
)

# Probability that a newly added op must output a vector.  Callers may
# override this (see `evolve_alphas`).
VECTOR_OPS_BIAS = 0.0

# Weighting logic centralized in utils.op_weight


if TYPE_CHECKING:
    from .alpha_framework_program import AlphaProgram # For type hint of self and return type


def mutate_program_logic(
    self_prog: AlphaProgram, # Instance of AlphaProgram
    feature_vars: Dict[str, TypeId],
    state_vars: Dict[str, TypeId],
    prob_add: float = 0.2, # Default values from original method
    prob_remove: float = 0.2,
    prob_change_op: float = 0.3,
    prob_change_inputs: float = 0.3,
    max_total_ops: int = 87,
    rng: Optional[np.random.Generator] = None,
    max_setup_ops: int = MAX_SETUP_OPS,
    max_predict_ops: int = MAX_PREDICT_OPS,
    max_update_ops: int = MAX_UPDATE_OPS,
    params: Optional[object] = None,
) -> AlphaProgram:
    """
    Core logic for AlphaProgram.mutate method.
    """
    rng = rng or np.random.default_rng()
    new_prog = self_prog.copy()
    from .utils import EvolutionParams  # local import to avoid cycles in type hints
    evo: Optional[EvolutionParams] = params if isinstance(params, EvolutionParams) else None
    local_vector_bias = VECTOR_OPS_BIAS if evo is None else evo.vector_ops_bias

    block_name_choices = ["predict"] * 6 + ["setup"] * 2 + ["update"] * 2
    chosen_block_name = rng.choice(block_name_choices)

    block_ref_map = {"setup": new_prog.setup, "predict": new_prog.predict_ops, "update": new_prog.update_ops}
    block_limit_map = {"setup": max_setup_ops, "predict": max_predict_ops, "update": max_update_ops}
    chosen_block_ops_list = block_ref_map[chosen_block_name]

    current_total_ops = sum(len(b) for b in block_ref_map.values())

    possible_mutations = []
    if current_total_ops < max_total_ops and len(chosen_block_ops_list) < block_limit_map[chosen_block_name]:
        possible_mutations.append("add")
    if len(chosen_block_ops_list) > (1 if chosen_block_name == "predict" else 0) :
        possible_mutations.append("remove")
    if len(chosen_block_ops_list) > 0:
        possible_mutations.extend(["change_op", "change_inputs"])

    if not possible_mutations:
        return new_prog  # No mutation possible
    mutation_type = rng.choice(possible_mutations)

    if mutation_type == "add":
        add_op_mutation(new_prog, chosen_block_name, feature_vars, state_vars, rng, local_vector_bias, evo)
    elif mutation_type == "remove":
        remove_op_mutation(new_prog, chosen_block_name, rng)
    elif mutation_type == "change_op":
        change_op_mutation(new_prog, chosen_block_name, rng, evo)
    elif mutation_type == "change_inputs":
        change_inputs_mutation(new_prog, chosen_block_name, rng, feature_vars, state_vars)

    _finalize_predict_tail(new_prog, feature_vars, state_vars, rng)

    # 1) remove dead / unreachable ops
    new_prog.prune()

    # 2) hard-cap each block to its max size
    new_prog.setup        = new_prog.setup[:max_setup_ops]
    new_prog.predict_ops  = new_prog.predict_ops[:max_predict_ops]
    new_prog.update_ops   = new_prog.update_ops[:max_update_ops]

    # 3) invalidate cached var-type map
    new_prog._vars_info_cache = None

    return new_prog


def crossover_program_logic(
    self_prog: AlphaProgram, # Instance of AlphaProgram (parent 1)
    other_prog: AlphaProgram, # Instance of AlphaProgram (parent 2)
    rng: Optional[np.random.Generator] = None,
    max_setup_ops: int = MAX_SETUP_OPS,
    max_predict_ops: int = MAX_PREDICT_OPS,
    max_update_ops: int = MAX_UPDATE_OPS,
    params: Optional[object] = None,
) -> AlphaProgram:
    """
    Core logic for AlphaProgram.crossover method.
    """
    rng = rng or np.random.default_rng()
    child = self_prog.copy()

    if child.predict_ops and other_prog.predict_ops:
        if child.predict_ops[-1].out == FINAL_PREDICTION_VECTOR_NAME:
            child.predict_ops.pop()

        other_predict_internal_ops = list(other_prog.predict_ops)
        if other_predict_internal_ops and other_predict_internal_ops[-1].out == FINAL_PREDICTION_VECTOR_NAME:
            other_predict_internal_ops.pop()

        len1, len2 = len(child.predict_ops), len(other_predict_internal_ops)
        if len1 > 0 and len2 > 0:
            pt1 = rng.integers(0, len1 + 1)
            pt2 = rng.integers(0, len2 + 1)
            new_predict_internal_ops = child.predict_ops[:pt1] + other_predict_internal_ops[pt2:]
            child.predict_ops = new_predict_internal_ops
        elif len2 > 0 :
            child.predict_ops = other_predict_internal_ops[rng.integers(0, len2+1):]

    elif other_prog.predict_ops: # Child's predict_ops was empty, other's was not.
        child.predict_ops = copy.deepcopy(other_prog.predict_ops) # Deepcopy from other
        if child.predict_ops and child.predict_ops[-1].out == FINAL_PREDICTION_VECTOR_NAME:
             child.predict_ops.pop() # Remove final op for generic fixup

    # Fixup logic for predict_ops
    temp_feature_vars = child._get_default_feature_vars() # Calling static method via instance
    temp_state_vars = {}

    if not child.predict_ops:
        default_vec_src = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t"
        child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_vec_src,)))
    else:
        vars_before_final_op_in_child = child.get_vars_at_point("predict", len(child.predict_ops) -1 , temp_feature_vars, temp_state_vars)
        last_op_in_child = child.predict_ops[-1]
        last_op_spec_in_child = OP_REGISTRY[last_op_in_child.opcode]

        actual_last_op_out_type_child = effective_out_type(
            last_op_spec_in_child, vars_before_final_op_in_child, last_op_in_child.inputs
        )

        if actual_last_op_out_type_child == "vector":
            child.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, last_op_in_child.opcode, last_op_in_child.inputs)
        else:
            vars_for_final_assign = {**vars_before_final_op_in_child, last_op_in_child.out: actual_last_op_out_type_child}
            available_vectors = [vn for vn, vt in vars_for_final_assign.items() if vt == "vector"]
            if not available_vectors:
                available_vectors = [fn for fn,ft in temp_feature_vars.items() if ft == "vector"]
                if not available_vectors and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
                    available_vectors = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES

            source_for_assign = rng.choice(available_vectors) if available_vectors else \
                                (CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t")
            child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_assign,)))

    child.prune()                                     # 1) drop dead code
    child.setup       = child.setup[:max_setup_ops]   # 2) enforce stage limits
    child.predict_ops = child.predict_ops[:max_predict_ops]
    child.update_ops  = child.update_ops[:max_update_ops]
    child._vars_info_cache = None                     # 3) clear cache

    return child


def _finalize_predict_tail(
    prog: "AlphaProgram",
    feature_vars: Dict[str, TypeId],
    state_vars: Dict[str, TypeId],
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Ensure predict block ends with a proper vector and repair if needed."""
    rng = rng or np.random.default_rng()
    if prog.predict_ops:
        last_op = prog.predict_ops[-1]
        last_op_spec = OP_REGISTRY[last_op.opcode]

        vars_before_last = prog.get_vars_at_point("predict", len(prog.predict_ops)-1, feature_vars, state_vars)
        actual_last_op_out_type = effective_out_type(
            last_op_spec, vars_before_last, last_op.inputs
        )

        if last_op.out != FINAL_PREDICTION_VECTOR_NAME or actual_last_op_out_type != "vector":
            if actual_last_op_out_type == "vector":
                prog.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, last_op.opcode, last_op.inputs)
            else:
                vars_for_final_fix = {**vars_before_last, last_op.out: actual_last_op_out_type}
                source_for_final_fix = pick_vector_fallback(vars_for_final_fix, feature_vars, rng=rng)
                prog.predict_ops.append(
                    Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_final_fix,))
                )
    else:
        default_feat_vec = pick_vector_fallback({}, feature_vars, rng=rng)
        prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_feat_vec,)))

    # — enforce that predict_ops[-1] is not a pure scalar aggregator —
    last = prog.predict_ops[-1]
    spec = OP_REGISTRY[last.opcode]
    if spec.is_cross_sectional_aggregator:
        prog.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, "cs_rank", ("vol10_t",))


def add_op_mutation(
    prog: "AlphaProgram",
    block_name: str,
    feature_vars: Dict[str, TypeId],
    state_vars: Dict[str, TypeId],
    rng: np.random.Generator,
    vector_bias: float,
    params: Optional[object] = None,
) -> None:
    """Insert a new op into the chosen block respecting typing and limits."""
    ops_list = prog.setup if block_name == "setup" else prog.predict_ops if block_name == "predict" else prog.update_ops
    insertion_idx = int(rng.integers(0, len(ops_list) + 1))
    vars_at_insertion = prog.get_vars_at_point(block_name, insertion_idx, feature_vars, state_vars)

    temp_current_vars = vars_at_insertion.copy()
    candidate_ops_for_add = []
    for op_n, op_s in OP_REGISTRY.items():
        if (
            block_name == "predict"
            and insertion_idx < len(ops_list)
            and op_s.out_type == "vector"
            and prog.predict_ops
            and (
                insertion_idx == len(ops_list) - 1
                and ops_list[insertion_idx].out == FINAL_PREDICTION_VECTOR_NAME
            )
        ):
            pass
        elif (
            op_n == "assign_vector" and block_name == "predict" and insertion_idx < len(ops_list)
        ):
            continue

        if rng.random() < vector_bias and op_s.out_type != "vector":
            continue

        formable = True
        temp_inputs_sources = []
        for req_t in op_s.in_types:
            cands = select_var_candidates(
                temp_current_vars,
                req_t,
                allow_elementwise_scalar_promotion=(op_s.is_elementwise and req_t == "scalar"),
            )
            if not cands:
                formable = False
                break
            temp_inputs_sources.append(cands)
        if formable:
            candidate_ops_for_add.append((op_n, op_s, temp_inputs_sources))

    if not candidate_ops_for_add:
        return

    weights = np.array([
        op_weight(op_n, is_predict=(block_name == "predict")) for op_n, _, _ in candidate_ops_for_add
    ], dtype=float)
    if np.all(weights <= 0) or np.isnan(weights).any():
        choice_index = int(rng.integers(len(candidate_ops_for_add)))
    else:
        weights = weights / weights.sum()
        choice_index = int(rng.choice(len(candidate_ops_for_add), p=weights))
    sel_op_n, sel_op_s, sel_sources = candidate_ops_for_add[choice_index]
    chosen_ins = tuple(rng.choice(s_list) for s_list in sel_sources)
    actual_out_t = effective_out_type(sel_op_s, temp_current_vars, chosen_ins)
    out_n = (
        FINAL_PREDICTION_VECTOR_NAME
        if (block_name == "predict" and insertion_idx == len(ops_list) and actual_out_t == "vector")
        else temp_name(actual_out_t, rng=rng)
    )
    ops_list.insert(insertion_idx, Op(out_n, sel_op_n, chosen_ins))


def remove_op_mutation(
    prog: "AlphaProgram",
    block_name: str,
    rng: np.random.Generator,
) -> None:
    ops_list = prog.setup if block_name == "setup" else prog.predict_ops if block_name == "predict" else prog.update_ops
    if not ops_list:
        return
    idx_to_remove = int(rng.integers(0, len(ops_list)))
    is_final_pred_op_targeted = (
        block_name == "predict" and ops_list[idx_to_remove].out == FINAL_PREDICTION_VECTOR_NAME and idx_to_remove == len(ops_list) - 1
    )
    if is_final_pred_op_targeted and len(ops_list) == 1:
        return
    ops_list.pop(idx_to_remove)


def change_op_mutation(
    prog: "AlphaProgram",
    block_name: str,
    rng: np.random.Generator,
    params: Optional[object] = None,
) -> None:
    ops_list = prog.setup if block_name == "setup" else prog.predict_ops if block_name == "predict" else prog.update_ops
    if not ops_list:
        return
    op_idx_to_change = int(rng.integers(0, len(ops_list)))
    original_op = ops_list[op_idx_to_change]
    is_final_predict_op = block_name == "predict" and original_op.out == FINAL_PREDICTION_VECTOR_NAME and op_idx_to_change == len(ops_list) - 1
    compatible = []
    for op_n, op_s in OP_REGISTRY.items():
        if len(op_s.in_types) == len(original_op.inputs) and op_n != original_op.opcode:
            if is_final_predict_op:
                if op_s.out_type == "vector" or (op_s.is_elementwise and op_s.out_type == "scalar"):
                    compatible.append(op_n)
            else:
                compatible.append(op_n)
    if not compatible:
        return
    weights = np.array([op_weight(op_n, is_predict=(block_name == "predict")) for op_n in compatible], dtype=float)
    if np.all(weights <= 0) or np.isnan(weights).any():
        new_opcode = str(rng.choice(compatible))
    else:
        weights = weights / weights.sum()
        new_opcode = str(rng.choice(compatible, p=weights))
    ops_list[op_idx_to_change] = Op(original_op.out, new_opcode, original_op.inputs)


def change_inputs_mutation(
    prog: "AlphaProgram",
    block_name: str,
    rng: np.random.Generator,
    feature_vars: Dict[str, TypeId],
    state_vars: Dict[str, TypeId],
) -> None:
    ops_list = prog.setup if block_name == "setup" else prog.predict_ops if block_name == "predict" else prog.update_ops
    if not ops_list:
        return
    op_idx_to_change = int(rng.integers(0, len(ops_list)))
    op_to_mutate = ops_list[op_idx_to_change]
    if not op_to_mutate.inputs:
        return
    vars_at_op = prog.get_vars_at_point(block_name, op_idx_to_change, feature_vars, state_vars)
    input_idx_to_change = int(rng.integers(0, len(op_to_mutate.inputs)))
    original_input_name = op_to_mutate.inputs[input_idx_to_change]
    spec = OP_REGISTRY[op_to_mutate.opcode]
    required_type = spec.in_types[input_idx_to_change]
    eligible = select_var_candidates(
        vars_at_op,
        required_type,
        allow_elementwise_scalar_promotion=(spec.is_elementwise and required_type == "scalar"),
        exclude=[original_input_name],
    )
    if not eligible:
        return
    new_input_name = str(rng.choice(eligible))
    new_inputs = list(op_to_mutate.inputs)
    new_inputs[input_idx_to_change] = new_input_name
    ops_list[op_idx_to_change] = Op(op_to_mutate.out, op_to_mutate.opcode, tuple(new_inputs))
