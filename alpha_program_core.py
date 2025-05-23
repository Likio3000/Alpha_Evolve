from __future__ import annotations

"""alpha_program_core.py
========================================
Fully‑featured *instruction‑list* representation of an alpha program that
matches the usage expected by **evolve_alphas.py**:

* `AlphaProgram.random_program()` – class‑method seed generator
* `mutate()` / `crossover()` / `copy()` – evolutionary operators
* Cheap `size`, `depth` (≈ longest dependency chain), `to_string()` helpers
* Robust `fingerprint` for duplicate filtering cache

The numerical operator registry now better supports cross-sectional operations.

**Consultant-driven Changes (May 2024):**
*   In `AlphaProgram.random_program()`:
    *   The final operation in the 'predict' block is no longer forced to be `assign_vector`.
    *   It now attempts to select any operation that results in a 'vector' type.
    *   If no such suitable operation can be formed, it falls back to `assign_vector`
        from an existing vector variable to ensure program validity.
*   Removed `const_0` from `SCALAR_FEATURE_NAMES` to avoid `0 ** x` issues and
    force GP to combine real values.
"""

from dataclasses import dataclass, field, replace
import copy
import hashlib
import json
import random
import textwrap 
from typing import Callable, Dict, List, Tuple, Literal, Sequence, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# 1 ‑‑ Type system
# ---------------------------------------------------------------------------

TypeId = Literal["scalar", "vector", "matrix"]

CROSS_SECTIONAL_FEATURE_VECTOR_NAMES = [
    "opens_t", "highs_t", "lows_t", "closes_t", "volumes_t", "ranges_t",
    "ma5_t", "ma10_t", "ma20_t", "ma30_t",
    "vol5_t", "vol10_t", "vol20_t", "vol30_t"
]
# MODIFICATION: Removed const_0
SCALAR_FEATURE_NAMES = ["const_1", "const_neg_1"]


FINAL_PREDICTION_VECTOR_NAME = "s1_predictions_vector"


@dataclass
class OpSpec:
    func: Callable
    in_types: Tuple[TypeId, ...]
    out_type: TypeId
    is_cross_sectional_aggregator: bool = False 
    is_elementwise: bool = False 


OP_REGISTRY: Dict[str, OpSpec] = {}


def register_op(name: str, *, in_types: Tuple[TypeId, ...], out: TypeId,
                is_cross_sectional_aggregator: bool = False,
                is_elementwise: bool = False):
    def _wrapper(fn: Callable):
        if name in OP_REGISTRY:
            raise KeyError(f"opcode '{name}' registered twice")
        OP_REGISTRY[name] = OpSpec(fn, in_types, out, is_cross_sectional_aggregator, is_elementwise)
        return fn
    return _wrapper


# ---------------------------------------------------------------------------
# 2 ‑‑ Primitive operators
# ---------------------------------------------------------------------------

@register_op("add", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _add(a, b): return a + b

@register_op("sub", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _sub(a, b): return a - b

@register_op("mul", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _mul(a, b): return a * b

@register_op("div", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _div(a, b):
    b_arr = np.asarray(b)
    a_arr_for_out = np.asarray(a) if not np.isscalar(a) else a 
    out_shape_ref = a_arr_for_out if hasattr(a_arr_for_out, 'shape') else b_arr

    res = np.divide(a, b_arr, 
                    out=np.zeros_like(out_shape_ref, dtype=float) if hasattr(out_shape_ref, 'shape') else 0.0, 
                    where=np.abs(b_arr) > 1e-9)
    if np.isscalar(a) and np.isscalar(b): return res.item() if isinstance(res, np.ndarray) else res
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
    return np.log(np.maximum(np.asarray(a), 1e-9)) 

@register_op("sqrt", in_types=("scalar",), out="scalar", is_elementwise=True)
def _sqrt(a): return np.sqrt(np.maximum(np.asarray(a), 0)) 

@register_op("power", in_types=("scalar", "scalar"), out="scalar", is_elementwise=True)
def _power(a, b):
    a_arr = np.asarray(a, dtype=float)
    b_is_scalar = np.isscalar(b)
    b_val = b if b_is_scalar else np.asarray(b, dtype=float) # Keep b as scalar if it is for specific checks

    if b_is_scalar:
        if b_val == 2: return np.square(a_arr)
        if b_val == 1: return a_arr
        if b_val == 0: return np.ones_like(a_arr, dtype=float) 
        if b_val == 0.5: return np.sqrt(np.maximum(a_arr, 0.0)) 
        if b_val == -1: return _div(1.0, a_arr) 
        
        # Handle a_arr being effectively zero
        # This check should be performed carefully if a_arr can be an array
        is_a_zero = np.all(np.abs(a_arr) < 1e-9) if isinstance(a_arr, np.ndarray) else (abs(a_arr) < 1e-9)

        if is_a_zero:
            if b_val > 0: return np.zeros_like(a_arr, dtype=float) 
            if b_val == 0: return np.ones_like(a_arr, dtype=float) 
            # For 0^negative, result is inf. Sign depends on a_arr if it's an array of zeros.
            # If a_arr is scalar 0, sign is typically positive.
            # Let's return positive infinity for scalar 0, and element-wise for array a_arr
            if isinstance(a_arr, np.ndarray):
                return np.copysign(np.full_like(a_arr, np.inf, dtype=float), a_arr)
            else: # a_arr is scalar zero
                return np.inf # Or np.copysign(np.inf, a_arr) if -0.0 matters

        is_b_integer = (isinstance(b_val, (int, float)) and b_val == round(b_val))
        # If a_arr is an array and any element is negative with fractional b_val
        if isinstance(a_arr, np.ndarray) and np.any(a_arr < 0) and not is_b_integer:
             return np.sign(a_arr) * np.power(np.abs(a_arr), b_val) # Signed power for array a
        elif not isinstance(a_arr, np.ndarray) and a_arr < 0 and not is_b_integer: # scalar a is negative
             return np.sign(a_arr) * np.power(np.abs(a_arr), b_val)
        else: 
            return np.power(a_arr, b_val)

    else: # b is an array (b_val is array here)
        b_arr_exponent = b_val 
        result = np.full_like(a_arr, np.nan, dtype=float) 

        mask_a_zero = np.abs(a_arr) < 1e-9
        # Must use b_arr_exponent here as b is an array
        result[mask_a_zero & (b_arr_exponent > 0)] = 0.0
        result[mask_a_zero & (b_arr_exponent == 0)] = 1.0
        # For 0^negative with array b, we need to handle signs carefully.
        # np.copysign is useful.
        zero_neg_exp_mask = mask_a_zero & (b_arr_exponent < 0)
        if np.any(zero_neg_exp_mask): # ensure indices are valid before assignment
             result[zero_neg_exp_mask] = np.copysign(np.inf, a_arr[zero_neg_exp_mask])


        mask_safe_power = (~mask_a_zero) & ((a_arr > 0) | (np.isclose(b_arr_exponent, np.round(b_arr_exponent))))
        if np.any(mask_safe_power):
            result[mask_safe_power] = np.power(a_arr[mask_safe_power], b_arr_exponent[mask_safe_power])

        mask_signed_power = (~mask_a_zero) & (a_arr < 0) & (~np.isclose(b_arr_exponent, np.round(b_arr_exponent)))
        if np.any(mask_signed_power):
            result[mask_signed_power] = np.sign(a_arr[mask_signed_power]) * \
                                        np.power(np.abs(a_arr[mask_signed_power]), b_arr_exponent[mask_signed_power])
        
        return np.nan_to_num(result, nan=0.0, posinf=np.finfo(float).max/2, neginf=np.finfo(float).min/2)


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
    if v.size <= 1: return np.zeros_like(v)
    temp = v.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(v))
    return (ranks / (len(v) - 1 + 1e-9)) * 2.0 - 1.0

@register_op("cs_demean", in_types=("vector",), out="vector")
def _cs_demean(v): return v - (_cs_mean(v) if v.size > 0 else 0.0)

@register_op("vec_add_scalar", in_types=("vector", "scalar"), out="vector")
def _vec_add_scalar(v, s): return v + s

@register_op("vec_mul_scalar", in_types=("vector", "scalar"), out="vector")
def _vec_mul_scalar(v, s): return v * s

@register_op("vec_div_scalar", in_types=("vector", "scalar"), out="vector")
def _vec_div_scalar(v, s): return v / (s if np.abs(s) > 1e-9 else np.copysign(1e-9, s)) 


# Matrix ops
@register_op("matmul_mv", in_types=("matrix", "vector"), out="vector")
def _matmul_mv(m, v): return m @ v

@register_op("matmul_mm", in_types=("matrix", "matrix"), out="matrix")
def _matmul_mm(m1, m2): return m1 @ m2

@register_op("transpose", in_types=("matrix",), out="matrix")
def _transpose(m): return m.T

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

# ---------------------------------------------------------------------------
# 3 ‑‑ Instruction & program container
# ---------------------------------------------------------------------------

@dataclass
class Op:
    out: str
    opcode: str
    inputs: Tuple[str, ...]

    def execute(self, buf: Dict[str, np.ndarray], n_stocks: int):
        spec = OP_REGISTRY[self.opcode]
        
        processed_args = []
        for i, in_name in enumerate(self.inputs):
            arg_val = buf.get(in_name) 
            if arg_val is None:
                raise KeyError(f"Op '{self.opcode}' input variable '{in_name}' not found in buffer. Buffer keys: {list(buf.keys())}")

            expected_type = spec.in_types[i]

            if expected_type == "scalar":
                if isinstance(arg_val, np.ndarray): 
                    if arg_val.size == 1: processed_args.append(arg_val.item())
                    elif spec.is_elementwise and arg_val.ndim == 1:
                         processed_args.append(arg_val) 
                    else:
                        processed_args.append(np.mean(arg_val) if arg_val.size > 0 else 0.0) 
                elif np.isscalar(arg_val): 
                    processed_args.append(float(arg_val))
                else:
                    raise TypeError(f"Op '{self.opcode}' input '{in_name}' expected scalar, got {type(arg_val)} value {arg_val}")

            elif expected_type == "vector":
                if np.isscalar(arg_val): 
                    processed_args.append(np.full(n_stocks, float(arg_val)))
                elif isinstance(arg_val, np.ndarray) and arg_val.ndim == 1:
                    if arg_val.shape[0] != n_stocks and not spec.is_cross_sectional_aggregator: 
                        if arg_val.size == 1:
                            processed_args.append(np.full(n_stocks, arg_val.item()))
                        else: 
                            resized_arr = np.zeros(n_stocks, dtype=arg_val.dtype)
                            copy_len = min(len(arg_val), n_stocks)
                            resized_arr[:copy_len] = arg_val[:copy_len]
                            processed_args.append(resized_arr)
                    else:
                        processed_args.append(arg_val)
                else:
                    raise TypeError(f"Op '{self.opcode}' input '{in_name}' expected vector, got {type(arg_val)} value {arg_val}")
            elif expected_type == "matrix":
                if not (isinstance(arg_val, np.ndarray) and arg_val.ndim == 2):
                    raise TypeError(f"Op '{self.opcode}' input '{in_name}' expected matrix, got {type(arg_val)} value {arg_val}")
                processed_args.append(arg_val)
        
        result = spec.func(*processed_args)
        
        if spec.is_elementwise and spec.out_type == "scalar" and isinstance(result, np.ndarray) and result.ndim == 1:
            pass 
        elif spec.out_type == "vector" and np.isscalar(result):
            result = np.full(n_stocks, float(result))
        elif spec.out_type == "vector" and isinstance(result, np.ndarray) and result.ndim == 1 and result.shape[0] != n_stocks and not spec.is_cross_sectional_aggregator:
            if result.size == 1: result = np.full(n_stocks, result.item()) 
            else:
                resized_res = np.zeros(n_stocks, dtype=result.dtype)
                copy_len_res = min(len(result), n_stocks)
                resized_res[:copy_len_res] = result[:copy_len_res]
                result = resized_res

        buf[self.out] = result

    def __str__(self):
        return f"{self.out} = {self.opcode}({', '.join(self.inputs)})"


@dataclass
class AlphaProgram:
    setup: List[Op] = field(default_factory=list)
    predict_ops: List[Op] = field(default_factory=list)
    update_ops: List[Op] = field(default_factory=list)

    _vars_info_cache: Optional[Dict[str, Dict[str, TypeId]]] = field(default=None, repr=False, compare=False)


    def _trace_vars_for_block(self, ops_block: List[Op], initial_vars: Dict[str, TypeId]) -> Dict[str, TypeId]:
        current_vars = initial_vars.copy()
        for op_instance in ops_block:
            spec = OP_REGISTRY[op_instance.opcode]
            actual_out_type = spec.out_type
            if spec.is_elementwise and spec.out_type == "scalar":
                is_any_input_vector = False
                for i, in_name in enumerate(op_instance.inputs):
                    input_var_type = current_vars.get(in_name)
                    if input_var_type == "vector":
                        if spec.in_types[i] == "scalar": 
                            is_any_input_vector = True; break
                if is_any_input_vector:
                    actual_out_type = "vector"
            current_vars[op_instance.out] = actual_out_type
        return current_vars
    
    def get_vars_at_point(self, block_name: Literal["setup", "predict", "update"], op_index: int,
                          feature_vars: Dict[str, TypeId], state_vars: Dict[str, TypeId]) -> Dict[str, TypeId]:
        available_vars = {}
        base_vars = {**feature_vars, **state_vars}

        if block_name == "setup":
            available_vars = self._trace_vars_for_block(self.setup[:op_index], base_vars)
        elif block_name == "predict":
            vars_after_setup = self._trace_vars_for_block(self.setup, base_vars)
            available_vars = self._trace_vars_for_block(self.predict_ops[:op_index], vars_after_setup)
        elif block_name == "update":
            vars_after_setup = self._trace_vars_for_block(self.setup, base_vars)
            vars_after_predict = self._trace_vars_for_block(self.predict_ops, vars_after_setup)
            merged_context_before_update = {**vars_after_predict} 
            available_vars = self._trace_vars_for_block(self.update_ops[:op_index], merged_context_before_update )
        
        return {**base_vars, **available_vars}


    @classmethod
    def random_program(cls, feature_vars: Dict[str, TypeId], state_vars: Dict[str, TypeId],
                       max_total_ops: int = 32, rng: Optional[np.random.Generator] = None) -> "AlphaProgram":
        rng = rng or np.random.default_rng()
        prog = cls()
        
        n_predict_ops = max(1, int(max_total_ops * 0.7)) 
        n_setup_ops = int(max_total_ops * 0.15)
        n_update_ops = max(0, max_total_ops - n_predict_ops - n_setup_ops)

        tmp_idx = 0
        def new_tmp_name(type_hint: TypeId):
            prefix = "s" if type_hint == "scalar" else ("v" if type_hint == "vector" else "m")
            nonlocal tmp_idx
            tmp_idx += 1
            return f"{prefix}{tmp_idx}"

        def add_random_op_to_block(ops_block: List[Op], initial_vars_for_block: Dict[str, TypeId], 
                                   num_ops_to_add: int, is_predict_block: bool):
            
            current_block_vars = initial_vars_for_block.copy()

            for op_count in range(num_ops_to_add):
                possible_ops_specs = [] 
                for op_name, spec in OP_REGISTRY.items():
                    is_last_predict_op_iteration = is_predict_block and op_count == num_ops_to_add - 1
                    
                    if op_name == "assign_vector" and spec.out_type == "vector":
                        if not is_last_predict_op_iteration: 
                            continue 
                    
                    potential_input_sources_for_spec = []
                    formable = True
                    for req_type in spec.in_types:
                        candidates = [v_name for v_name, v_type in current_block_vars.items() if v_type == req_type]
                        if not candidates and req_type == "scalar" and spec.is_elementwise:
                            vec_candidates = [v_name for v_name, v_type in current_block_vars.items() if v_type == "vector"]
                            if vec_candidates: candidates.extend(vec_candidates)
                        if not candidates:
                            formable = False; break
                        potential_input_sources_for_spec.append(candidates)
                    
                    if formable:
                        possible_ops_specs.append((op_name, spec, potential_input_sources_for_spec))

                if not possible_ops_specs: break 

                selected_op_name, selected_spec, chosen_inputs, actual_out_type = None, None, None, None
                is_last_predict_op = is_predict_block and op_count == num_ops_to_add - 1

                if is_last_predict_op:
                    found_suitable_final_op = False
                    shuffled_possible_ops = list(possible_ops_specs) 
                    rng.shuffle(shuffled_possible_ops)

                    for op_n_cand, spec_cand, sources_list_cand in shuffled_possible_ops:
                        inputs_cand = tuple(rng.choice(s) for s in sources_list_cand) 
                        
                        current_actual_out_type_cand = spec_cand.out_type
                        if spec_cand.is_elementwise and spec_cand.out_type == "scalar":
                            if any(current_block_vars.get(inp_n) == "vector" for inp_n in inputs_cand):
                                current_actual_out_type_cand = "vector"
                        
                        if current_actual_out_type_cand == "vector": 
                            selected_op_name, selected_spec, chosen_inputs, actual_out_type = \
                                op_n_cand, spec_cand, inputs_cand, current_actual_out_type_cand
                            found_suitable_final_op = True
                            break 
                    
                    if not found_suitable_final_op:
                        available_true_vectors = [vn for vn, vt in current_block_vars.items() if vt == "vector"]
                        source_for_final_assign = None
                        if available_true_vectors:
                            source_for_final_assign = rng.choice(available_true_vectors)
                        else:
                            candidate_feature_vectors = [
                                fn for fn, ft in feature_vars.items() 
                                if ft == "vector" and fn in current_block_vars 
                            ]
                            if not candidate_feature_vectors:
                                 candidate_feature_vectors = [fn for fn, ft in feature_vars.items() if ft == "vector"]
                            if not candidate_feature_vectors:
                                if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES: 
                                    source_for_final_assign = rng.choice(CROSS_SECTIONAL_FEATURE_VECTOR_NAMES)
                                else:
                                    raise ValueError("CRITICAL: No vector features available for final assign_vector fallback.")
                            else:
                                source_for_final_assign = rng.choice(candidate_feature_vectors)
                        
                        selected_op_name = "assign_vector" 
                        selected_spec = OP_REGISTRY["assign_vector"]
                        chosen_inputs = (source_for_final_assign,)
                        actual_out_type = "vector"
                else: 
                    choice_index = rng.integers(len(possible_ops_specs))
                    op_n_cand, spec_cand, sources_list_cand = possible_ops_specs[choice_index]
                    
                    chosen_inputs_cand = tuple(rng.choice(s) for s in sources_list_cand)
                    actual_out_type_cand = spec_cand.out_type
                    if spec_cand.is_elementwise and spec_cand.out_type == "scalar":
                        if any(current_block_vars.get(inp_n) == "vector" for inp_n in chosen_inputs_cand):
                            actual_out_type_cand = "vector"
                    
                    selected_op_name, selected_spec, chosen_inputs, actual_out_type = \
                        op_n_cand, spec_cand, chosen_inputs_cand, actual_out_type_cand
                
                if is_last_predict_op: 
                    out_var_name = FINAL_PREDICTION_VECTOR_NAME
                    if actual_out_type != "vector":
                        out_var_name = FINAL_PREDICTION_VECTOR_NAME
                        selected_op_name = "assign_vector"
                        selected_spec = OP_REGISTRY["assign_vector"]
                        
                        cs_vec_names = [fn for fn, ft in feature_vars.items() if ft == "vector"]
                        if not cs_vec_names: cs_vec_names = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES 
                        if not cs_vec_names: 
                            if "opens_t" in initial_vars_for_block : cs_vec_names.append("opens_t") 
                            else: raise ValueError("Cannot find any vector source for emergency final assignment.")
                        
                        source_for_emergency_assign = rng.choice(cs_vec_names)
                        chosen_inputs = (source_for_emergency_assign,)
                        actual_out_type = "vector"
                else:
                    out_var_name = new_tmp_name(actual_out_type)
                
                ops_block.append(Op(out_var_name, selected_op_name, chosen_inputs))
                current_block_vars[out_var_name] = actual_out_type
        
        setup_initial = {**feature_vars, **state_vars}
        add_random_op_to_block(prog.setup, setup_initial, n_setup_ops, False)
        
        vars_after_setup = prog._trace_vars_for_block(prog.setup, setup_initial)
        predict_initial = {**vars_after_setup} 
        add_random_op_to_block(prog.predict_ops, predict_initial, n_predict_ops, True)

        vars_after_predict = prog._trace_vars_for_block(prog.predict_ops, predict_initial)
        update_initial = {**vars_after_predict}
        add_random_op_to_block(prog.update_ops, update_initial, n_update_ops, False)
        
        return prog

    def copy(self) -> "AlphaProgram":
        new_prog = AlphaProgram(
            setup=copy.deepcopy(self.setup),
            predict_ops=copy.deepcopy(self.predict_ops),
            update_ops=copy.deepcopy(self.update_ops)
        )
        return new_prog

    def mutate(self, feature_vars: Dict[str, TypeId], state_vars: Dict[str, TypeId],
               prob_add: float = 0.2, prob_remove: float = 0.2, 
               prob_change_op: float = 0.3, prob_change_inputs: float = 0.3, 
               max_total_ops: int = 48, rng: Optional[np.random.Generator] = None) -> "AlphaProgram":
        rng = rng or np.random.default_rng()
        new_prog = self.copy()

        block_name_choices = ["predict"] * 6 + ["setup"] * 2 + ["update"] * 2
        chosen_block_name = rng.choice(block_name_choices)
        
        block_ref_map = {"setup": new_prog.setup, "predict": new_prog.predict_ops, "update": new_prog.update_ops}
        chosen_block_ops_list = block_ref_map[chosen_block_name]

        current_total_ops = sum(len(b) for b in block_ref_map.values())
        
        possible_mutations = []
        if current_total_ops < max_total_ops: possible_mutations.append("add")
        if len(chosen_block_ops_list) > (1 if chosen_block_name == "predict" else 0) : 
            possible_mutations.append("remove")
        if len(chosen_block_ops_list) > 0:
            possible_mutations.extend(["change_op", "change_inputs"])
        
        if not possible_mutations: return new_prog
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

                formable = True
                temp_inputs_sources = []
                for req_t in op_s.in_types:
                    sources = [vn for vn, vt in temp_current_vars.items() if vt == req_t]
                    if not sources and req_t == "scalar" and op_s.is_elementwise:
                        sources = [vn for vn, vt in temp_current_vars.items() if vt == "vector"]
                    if not sources: formable=False; break
                    temp_inputs_sources.append(sources)
                if formable: candidate_ops_for_add.append((op_n, op_s, temp_inputs_sources))
            
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
                if chosen_block_name == "predict" and \
                   chosen_block_ops_list[idx_to_remove].out == FINAL_PREDICTION_VECTOR_NAME and \
                   idx_to_remove == len(chosen_block_ops_list) -1 :
                    pass 
                else:
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
            if not op_to_mutate.inputs: return new_prog 

            vars_at_op = new_prog.get_vars_at_point(chosen_block_name, op_idx_to_change, feature_vars, state_vars)
            input_idx_to_change = rng.integers(0, len(op_to_mutate.inputs))
            
            original_input_name = op_to_mutate.inputs[input_idx_to_change]
            spec_of_op_to_mutate = OP_REGISTRY[op_to_mutate.opcode] 
            required_type = spec_of_op_to_mutate.in_types[input_idx_to_change]

            candidates = [vn for vn, vt in vars_at_op.items() if vt == required_type and vn != original_input_name]
            if not candidates and required_type == "scalar" and spec_of_op_to_mutate.is_elementwise:
                candidates.extend([vn for vn, vt in vars_at_op.items() if vt == "vector" and vn != original_input_name])
            
            if candidates:
                new_input_name = rng.choice(candidates)
                new_inputs_tuple = list(op_to_mutate.inputs)
                new_inputs_tuple[input_idx_to_change] = new_input_name
                chosen_block_ops_list[op_idx_to_change] = Op(op_to_mutate.out, op_to_mutate.opcode, tuple(new_inputs_tuple))

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
                     
                    if not available_vectors: available_vectors = [fn for fn,ft in feature_vars.items() if ft == "vector"] 
                    
                    if available_vectors:
                        source_for_final_fix = rng.choice(available_vectors)
                        new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_final_fix,)))
        elif not new_prog.predict_ops : 
            default_feat_vec = next((vn for vn, vt in feature_vars.items() if vt == "vector"), None)
            if not default_feat_vec and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
                default_feat_vec = rng.choice(CROSS_SECTIONAL_FEATURE_VECTOR_NAMES)
            elif not default_feat_vec: 
                default_feat_vec = "opens_t" 

            if default_feat_vec: # Ensure default_feat_vec is not None before using
                 new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_feat_vec,)))


        new_prog._vars_info_cache = None 
        return new_prog

    def crossover(self, other: "AlphaProgram", rng: Optional[np.random.Generator] = None) -> "AlphaProgram":
        rng = rng or np.random.default_rng()
        child = self.copy() 

        if child.predict_ops and other.predict_ops:
            child_final_op_obj = None
            if child.predict_ops[-1].out == FINAL_PREDICTION_VECTOR_NAME:
                child_final_op_obj = child.predict_ops.pop()

            other_predict_internal_ops = list(other.predict_ops) 
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
            
            temp_feature_vars = AlphaProgram._get_default_feature_vars() 
            temp_state_vars = {} 
            vars_for_final_op = child.get_vars_at_point("predict", len(child.predict_ops), temp_feature_vars, temp_state_vars)

            available_vectors = [vn for vn, vt in vars_for_final_op.items() if vt == "vector"]
            if not available_vectors: 
                available_vectors = [fn for fn,ft in temp_feature_vars.items() if ft == "vector"]
            
            final_op_to_add = None
            if child_final_op_obj: 
                original_final_opcode_str = child_final_op_obj.opcode # Get the string name
                final_op_spec = OP_REGISTRY[original_final_opcode_str] # Get spec
                
                can_produce_vector = final_op_spec.out_type == "vector" or \
                                     (final_op_spec.is_elementwise and final_op_spec.out_type == "scalar") 

                if can_produce_vector:
                    # CORRECTED: Compare the string opcode with "assign_vector"
                    if original_final_opcode_str == "assign_vector": 
                        if available_vectors:
                            final_op_to_add = Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (rng.choice(available_vectors),))


            if not final_op_to_add: 
                if available_vectors:
                    source_for_final = rng.choice(available_vectors)
                    final_op_to_add = Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_final,))
                else: 
                    default_cs_vec_name = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t" 
                    final_op_to_add = Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_cs_vec_name,))
            
            child.predict_ops.append(final_op_to_add)

        elif other.predict_ops: 
            child.predict_ops = copy.deepcopy(other.predict_ops) 

        child._vars_info_cache = None
        return child

    @staticmethod
    def _get_default_feature_vars() -> Dict[str, TypeId]:
        default_vars = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
        default_vars.update({name: "scalar" for name in SCALAR_FEATURE_NAMES})
        if "const_1" not in default_vars: default_vars["const_1"] = "scalar"
        if "const_neg_1" not in default_vars: default_vars["const_neg_1"] = "scalar"
        return default_vars

    def new_state(self) -> Dict[str, Union[np.ndarray, float]]:
        return {}

    def eval(self, features_at_t: Dict[str, Union[np.ndarray, float]], 
               state: Dict[str, Union[np.ndarray, float]],
               n_stocks: int) -> np.ndarray:
        self._vars_info_cache = None 
        
        buf: Dict[str, Union[np.ndarray, float]] = {**features_at_t, **state}

        for op_instance in self.setup:
            op_instance.execute(buf, n_stocks)
        
        if not self.predict_ops:
            return np.full(n_stocks, np.nan) 
            
        for op_instance in self.predict_ops:
            try:
                op_instance.execute(buf, n_stocks)
            except Exception: 
                return np.full(n_stocks, np.nan)


        if FINAL_PREDICTION_VECTOR_NAME not in buf:
            return np.full(n_stocks, np.nan)

        s1_predictions_val = buf[FINAL_PREDICTION_VECTOR_NAME]

        if np.isscalar(s1_predictions_val): 
            s1_predictions_val = np.full(n_stocks, float(s1_predictions_val))
        
        if not isinstance(s1_predictions_val, np.ndarray) or s1_predictions_val.ndim != 1 or s1_predictions_val.shape[0] != n_stocks:
            return np.full(n_stocks, np.nan) 

        initial_state_keys = set(state.keys())
        vars_defined_in_update = set()

        for op_instance in self.update_ops:
            try:
                op_instance.execute(buf, n_stocks)
                vars_defined_in_update.add(op_instance.out)
            except Exception: 
                return np.full(n_stocks, np.nan) 

        for key in list(state.keys()): 
            if key in buf and key not in features_at_t:
                state[key] = buf[key]

        for key in vars_defined_in_update:
            if key not in features_at_t: 
                 state[key] = buf[key]

        return np.nan_to_num(s1_predictions_val.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


    @property
    def size(self) -> int:
        return len(self.setup) + len(self.predict_ops) + len(self.update_ops)

    def to_string(self, max_len: int = 120) -> str:
        txt_parts = []
        if self.setup: txt_parts.append(f"S[{';'.join(map(str, self.setup))}]")
        if self.predict_ops: txt_parts.append(f"P[{';'.join(map(str, self.predict_ops))}]")
        if self.update_ops: txt_parts.append(f"U[{';'.join(map(str, self.update_ops))}]")
        
        full_txt = " >> ".join(txt_parts)
        return textwrap.shorten(full_txt, width=max_len, placeholder="...")

    @property
    def fingerprint(self) -> str:
        serial = {
            "setup": [(o.out, o.opcode, o.inputs) for o in self.setup],
            "predict": [(o.out, o.opcode, o.inputs) for o in self.predict_ops],
            "update": [(o.out, o.opcode, o.inputs) for o in self.update_ops],
        }
        return hashlib.sha1(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest()