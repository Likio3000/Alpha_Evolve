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

# Probability that a newly added op must output a vector.  Callers may
# override this (see `evolve_alphas`).
VECTOR_OPS_BIAS = 0.0


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
) -> AlphaProgram:
    """
    Core logic for AlphaProgram.mutate method.
    """
    rng = rng or np.random.default_rng()
    new_prog = self_prog.copy()

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
        insertion_idx = rng.integers(0, len(chosen_block_ops_list) + 1)
        vars_at_insertion = new_prog.get_vars_at_point(chosen_block_name, insertion_idx, feature_vars, state_vars)

        temp_current_vars = vars_at_insertion.copy()
        candidate_ops_for_add = []
        for op_n, op_s in OP_REGISTRY.items():
            if chosen_block_name == "predict" and insertion_idx < len(chosen_block_ops_list) and \
               op_s.out_type == "vector" and new_prog.predict_ops and \
               (insertion_idx == len(chosen_block_ops_list) -1 and chosen_block_ops_list[insertion_idx].out == FINAL_PREDICTION_VECTOR_NAME ) :
                pass
            elif op_n == "assign_vector" and chosen_block_name == "predict" and insertion_idx < len(chosen_block_ops_list):
                 continue

            # ─── bias towards ops that output vectors ───
            if rng.random() < VECTOR_OPS_BIAS and op_s.out_type != "vector":
                continue

            formable = True
            temp_inputs_sources = []
            for req_t in op_s.in_types:
                current_type_candidates_add = []
                if req_t == "scalar":
                    const_scalars_add = []
                    other_scalars_add = []
                    for vn_add, vt_add in temp_current_vars.items():
                        if vt_add == "scalar":
                            if vn_add in SCALAR_FEATURE_NAMES:
                                const_scalars_add.append(vn_add)
                            else:
                                other_scalars_add.append(vn_add)
                    if other_scalars_add:
                        current_type_candidates_add.extend(other_scalars_add * 3)
                        current_type_candidates_add.extend(const_scalars_add)
                    elif const_scalars_add:
                        current_type_candidates_add.extend(const_scalars_add)

                    if not current_type_candidates_add and op_s.is_elementwise:
                        vec_opts_add = [vn_add for vn_add, vt_add in temp_current_vars.items() if vt_add == "vector"]
                        if vec_opts_add:
                            current_type_candidates_add.extend(vec_opts_add)
                else: # vector or matrix
                    current_type_candidates_add = [vn_add for vn_add, vt_add in temp_current_vars.items() if vt_add == req_t]

                if not current_type_candidates_add:
                    formable = False
                    break
                temp_inputs_sources.append(current_type_candidates_add)
            if formable:
                candidate_ops_for_add.append((op_n, op_s, temp_inputs_sources))

        if candidate_ops_for_add:
            choice_index = rng.integers(len(candidate_ops_for_add))
            sel_op_n, sel_op_s, sel_sources = candidate_ops_for_add[choice_index]
            chosen_ins = tuple(rng.choice(s_list) for s_list in sel_sources)

            actual_out_t = sel_op_s.out_type
            if sel_op_s.is_elementwise and sel_op_s.out_type == "scalar":
                 if any(temp_current_vars.get(inp_n) == "vector" for inp_n in chosen_ins):
                    actual_out_t = "vector"

            out_n = ""
            if chosen_block_name == "predict" and insertion_idx == len(chosen_block_ops_list) and actual_out_t == "vector":
                out_n = FINAL_PREDICTION_VECTOR_NAME
            else:
                tmp_idx_mut = rng.integers(10000, 20000)
                out_n = f"m{tmp_idx_mut}_{actual_out_t[0]}"

            new_op_to_insert = Op(out_n, sel_op_n, chosen_ins)
            chosen_block_ops_list.insert(insertion_idx, new_op_to_insert)

    elif mutation_type == "remove":
        if chosen_block_ops_list:
            idx_to_remove = rng.integers(0, len(chosen_block_ops_list))
            is_final_pred_op_targeted = chosen_block_name == "predict" and \
                                       chosen_block_ops_list[idx_to_remove].out == FINAL_PREDICTION_VECTOR_NAME and \
                                       idx_to_remove == len(chosen_block_ops_list) - 1

            can_remove = True
            if is_final_pred_op_targeted and len(chosen_block_ops_list) == 1:
                can_remove = False

            if can_remove and not is_final_pred_op_targeted :
                chosen_block_ops_list.pop(idx_to_remove)
            elif can_remove and is_final_pred_op_targeted and len(chosen_block_ops_list)>1:
                 chosen_block_ops_list.pop(idx_to_remove)


    elif mutation_type == "change_op" and chosen_block_ops_list:
        op_idx_to_change = rng.integers(0, len(chosen_block_ops_list))
        original_op = chosen_block_ops_list[op_idx_to_change]

        is_final_predict_op = chosen_block_name == "predict" and \
                              original_op.out == FINAL_PREDICTION_VECTOR_NAME and \
                              op_idx_to_change == len(chosen_block_ops_list) -1

        compatible_ops = []
        for op_n, op_s in OP_REGISTRY.items():
             if len(op_s.in_types) == len(original_op.inputs) and op_n != original_op.opcode:
                if is_final_predict_op:
                    if op_s.out_type == "vector" or (op_s.is_elementwise and op_s.out_type == "scalar"):
                         compatible_ops.append(op_n)
                else:
                    compatible_ops.append(op_n)

        if compatible_ops:
            new_opcode = rng.choice(compatible_ops)
            chosen_block_ops_list[op_idx_to_change] = Op(original_op.out, new_opcode, original_op.inputs)

    elif mutation_type == "change_inputs" and chosen_block_ops_list:
        op_idx_to_change = rng.integers(0, len(chosen_block_ops_list))
        op_to_mutate = chosen_block_ops_list[op_idx_to_change]
        if not op_to_mutate.inputs:
            return new_prog

        vars_at_op = new_prog.get_vars_at_point(chosen_block_name, op_idx_to_change, feature_vars, state_vars)
        input_idx_to_change = rng.integers(0, len(op_to_mutate.inputs))

        original_input_name = op_to_mutate.inputs[input_idx_to_change]
        spec_of_op_to_mutate = OP_REGISTRY[op_to_mutate.opcode]
        required_type = spec_of_op_to_mutate.in_types[input_idx_to_change]

        eligible_candidates = []
        if required_type == "scalar":
            const_scalars_options = []
            other_scalars_options = []
            for vn, vt in vars_at_op.items():
                if vn == original_input_name:
                    continue
                if vt == "scalar":
                    if vn in SCALAR_FEATURE_NAMES:
                        const_scalars_options.append(vn)
                    else:
                        other_scalars_options.append(vn)

            if other_scalars_options:
                eligible_candidates.extend(other_scalars_options * 3)
                eligible_candidates.extend(const_scalars_options)
            elif const_scalars_options:
                eligible_candidates.extend(const_scalars_options)

            if not eligible_candidates and spec_of_op_to_mutate.is_elementwise:
                vec_options_for_scalar_slot = [vn for vn, vt in vars_at_op.items() if vt == "vector" and vn != original_input_name]
                if vec_options_for_scalar_slot:
                    eligible_candidates.extend(vec_options_for_scalar_slot)
        else: # For "vector" or "matrix"
            eligible_candidates = [vn for vn, vt in vars_at_op.items() if vt == required_type and vn != original_input_name]

        if eligible_candidates:
            new_input_name = rng.choice(eligible_candidates)
            new_inputs_tuple = list(op_to_mutate.inputs)
            new_inputs_tuple[input_idx_to_change] = new_input_name
            chosen_block_ops_list[op_idx_to_change] = Op(op_to_mutate.out, op_to_mutate.opcode, tuple(new_inputs_tuple))

    # Ensure predict block ends with a vector named FINAL_PREDICTION_VECTOR_NAME
    if new_prog.predict_ops:
        last_op = new_prog.predict_ops[-1]
        last_op_spec = OP_REGISTRY[last_op.opcode]

        vars_before_last = new_prog.get_vars_at_point("predict", len(new_prog.predict_ops)-1, feature_vars, state_vars)
        actual_last_op_out_type = last_op_spec.out_type
        if last_op_spec.is_elementwise and last_op_spec.out_type == "scalar":
            if any(vars_before_last.get(inp_n) == "vector" for inp_n in last_op.inputs):
                 actual_last_op_out_type = "vector"

        if last_op.out != FINAL_PREDICTION_VECTOR_NAME or actual_last_op_out_type != "vector":
            if actual_last_op_out_type == "vector":
                 new_prog.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, last_op.opcode, last_op.inputs)
            else:
                vars_for_final_fix = {**vars_before_last, last_op.out: actual_last_op_out_type}
                available_vectors = [vn for vn, vt in vars_for_final_fix.items() if vt == "vector"]

                if not available_vectors:
                    available_vectors = [fn for fn,ft in feature_vars.items() if ft == "vector"]
                    if not available_vectors and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
                         available_vectors = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES

                if available_vectors:
                    source_for_final_fix = rng.choice(available_vectors)
                    new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_final_fix,)))
                else:
                    default_feat_vec_fallback = "opens_t" if "opens_t" in feature_vars else \
                                              (CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "fallback_undefined_vector")
                    new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_feat_vec_fallback,)))

    elif not new_prog.predict_ops : # Predict block became empty
        default_feat_vec = next((vn for vn, vt in feature_vars.items() if vt == "vector"), None)
        if not default_feat_vec and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
            default_feat_vec = rng.choice(CROSS_SECTIONAL_FEATURE_VECTOR_NAMES)
        elif not default_feat_vec:
            default_feat_vec = "opens_t" # Absolute fallback

        if default_feat_vec:
             new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_feat_vec,)))

    # — enforce that predict_ops[-1] is not a pure scalar aggregator —
    last = new_prog.predict_ops[-1]
    spec = OP_REGISTRY[last.opcode]
    if spec.is_cross_sectional_aggregator:
        new_prog.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, "cs_rank", ("vol10_t",))

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

        actual_last_op_out_type_child = last_op_spec_in_child.out_type
        if last_op_spec_in_child.is_elementwise and last_op_spec_in_child.out_type == "scalar":
            if any(vars_before_final_op_in_child.get(inp_n) == "vector" for inp_n in last_op_in_child.inputs):
                actual_last_op_out_type_child = "vector"

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
