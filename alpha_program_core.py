from __future__ import annotations

"""alpha_program_core.py"""


from dataclasses import dataclass, field
import copy, hashlib, json, textwrap
from typing import Callable, Dict, List, Tuple, Literal, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# 1 ‑‑ Type system
# ---------------------------------------------------------------------------

TypeId = Literal["scalar", "vector", "matrix"]

CROSS_SECTIONAL_FEATURE_VECTOR_NAMES = [
    "opens_t", "highs_t", "lows_t", "closes_t", "volumes_t", "ranges_t",
    "ma5_t", "ma10_t", "ma20_t", "ma30_t",
    "vol5_t", "vol10_t", "vol20_t", "vol30_t",
]
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

###############################################################################
# 2 -- Primitive operators
###############################################################################

SAFE_MAX = 1e6         # single knob: absolute value clamp for *all* ops

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
    """
    Safe element-wise exponentiation.

    * Silences overflow / invalid / divide warnings from NumPy.
    * Cleans NaN/±inf → finite.
    * Hard-clips result to ±MAX_ABS so numbers never explode downstream.
    """
    MAX_ABS = 1e9
    a_arr = np.asarray(a, dtype=float)
    b_is_scalar = np.isscalar(b)
    b_val = b if b_is_scalar else np.asarray(b, dtype=float)

    # ------------------------------------------------------------------ #
    def _clip(x):
        return np.clip(x, -MAX_ABS, MAX_ABS)

    # ------------------------- scalar exponent ------------------------ #
    if b_is_scalar:
        if b_val == 2:   return _clip(np.square(a_arr))
        if b_val == 1:   return _clip(a_arr)
        if b_val == 0:   return np.ones_like(a_arr, dtype=float)
        if b_val == 0.5: return _clip(np.sqrt(np.maximum(a_arr, 0.0)))
        if b_val == -1:  return _clip(_div(1.0, a_arr))

        is_a_zero = np.all(np.abs(a_arr) < 1e-9) if isinstance(a_arr, np.ndarray) \
                    else abs(a_arr) < 1e-9
        if is_a_zero:
            if b_val > 0:  return np.zeros_like(a_arr, dtype=float)
            if b_val == 0: return np.ones_like(a_arr, dtype=float)
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

        mask0 = np.abs(a_arr) < 1e-9
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
@safe_op
def _vec_mul_scalar(v, s): return v * s

@register_op("vec_div_scalar", in_types=("vector", "scalar"), out="vector")
@safe_op
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
# 3 – Instruction container
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
    def random_program(
        cls,
        feature_vars: Dict[str, TypeId],
        state_vars: Dict[str, TypeId],
        max_total_ops: int = 32,
        rng: Optional[np.random.Generator] = None
    ) -> "AlphaProgram":
        """
        Build a random but type-correct AlphaProgram.

        May-2025 change
        ─────────────────
        * If we cannot construct a **vector-typed** final operation for the
          predict block we now **raise RuntimeError** instead of falling back
          to the old «assign_vector(<one feature vector>)» shortcut.  The
          caller (evolution loop) just retries and eventually gets a real one.
        """
        rng = rng or np.random.default_rng()
        prog = cls()

        # — split total op budget into three blocks —
        n_predict_ops = max(1, int(max_total_ops * 0.70))
        n_setup_ops   = int(max_total_ops * 0.15)
        n_update_ops  = max_total_ops - n_predict_ops - n_setup_ops

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
                        # <-- this is the NEW behaviour
                        raise RuntimeError(
                            "random_program(): could not find a vector-typed "
                            "operation for the final predict slot"
                        )
                else:
                    idx = rng.integers(len(candidates))      # ← pick index
                    chosen_name, chosen_spec, pools = candidates[idx]
                    chosen_ins = tuple(rng.choice(p) for p in pools)
                    out_t = chosen_spec.out_type
                    if chosen_spec.is_elementwise and out_t == "scalar":
                        if any(current[i] == "vector" for i in chosen_ins):
                            out_t = "vector"

                # 3. emit op
                out_name = (FINAL_PREDICTION_VECTOR_NAME
                            if last_predict_slot
                            else _new_tmp(out_t))
                block.append(Op(out_name, chosen_name, chosen_ins))
                current[out_name] = out_t

        # ────────────────── actually build the three blocks ──────────────────
        _add_ops(prog.setup,   {**feature_vars, **state_vars},           n_setup_ops,   False)
        after_setup = prog._trace_vars_for_block(prog.setup,
                                                 {**feature_vars, **state_vars})

        _add_ops(prog.predict_ops, after_setup,                          n_predict_ops, True)
        after_predict = prog._trace_vars_for_block(prog.predict_ops,
                                                   after_setup)

        _add_ops(prog.update_ops, after_predict,                         n_update_ops,  False)
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
                    current_type_candidates_add = []
                    if req_t == "scalar":
                        const_scalars_add = []
                        other_scalars_add = []
                        for vn_add, vt_add in temp_current_vars.items():
                            if vt_add == "scalar":
                                if vn_add in SCALAR_FEATURE_NAMES: const_scalars_add.append(vn_add)
                                else: other_scalars_add.append(vn_add)
                        if other_scalars_add:
                            current_type_candidates_add.extend(other_scalars_add * 3)
                            current_type_candidates_add.extend(const_scalars_add)
                        elif const_scalars_add:
                            current_type_candidates_add.extend(const_scalars_add)
                        
                        if not current_type_candidates_add and op_s.is_elementwise:
                            vec_opts_add = [vn_add for vn_add, vt_add in temp_current_vars.items() if vt_add == "vector"]
                            if vec_opts_add: current_type_candidates_add.extend(vec_opts_add)
                    else: # vector or matrix
                        current_type_candidates_add = [vn_add for vn_add, vt_add in temp_current_vars.items() if vt_add == req_t]

                    if not current_type_candidates_add: formable=False; break
                    temp_inputs_sources.append(current_type_candidates_add)
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
                # Prevent removing the final prediction op if it's the only one or named as such
                is_final_pred_op_targeted = chosen_block_name == "predict" and \
                                           chosen_block_ops_list[idx_to_remove].out == FINAL_PREDICTION_VECTOR_NAME and \
                                           idx_to_remove == len(chosen_block_ops_list) - 1
                
                can_remove = True
                if is_final_pred_op_targeted and len(chosen_block_ops_list) == 1:
                    can_remove = False # Don't remove if it's the only op in predict block and is the final output

                if can_remove and not is_final_pred_op_targeted : # Prefer not to remove the specifically named final op unless it's not the last
                    chosen_block_ops_list.pop(idx_to_remove)
                elif can_remove and is_final_pred_op_targeted and len(chosen_block_ops_list)>1: # If it is the last, but not only one
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
                        # Determine if this new op_s can produce a vector (directly or via elementwise promotion)
                        # This requires knowing the input types, which is tricky without full re-trace here.
                        # For simplicity, we allow ops that output vector, or elementwise scalar ops (as they *could* become vector)
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

            eligible_candidates = [] 
            if required_type == "scalar":
                const_scalars_options = []
                other_scalars_options = []
                for vn, vt in vars_at_op.items():
                    if vn == original_input_name: continue 
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
                new_input_name = rng.choice(eligible_candidates) # Corrected: use eligible_candidates
                new_inputs_tuple = list(op_to_mutate.inputs)
                new_inputs_tuple[input_idx_to_change] = new_input_name
                chosen_block_ops_list[op_idx_to_change] = Op(op_to_mutate.out, op_to_mutate.opcode, tuple(new_inputs_tuple))

        # Ensure predict block ends with a vector named FINAL_PREDICTION_VECTOR_NAME
        if new_prog.predict_ops:
            last_op = new_prog.predict_ops[-1]
            last_op_spec = OP_REGISTRY[last_op.opcode]
            
            # Get variables available *before* the last op executes
            vars_before_last = new_prog.get_vars_at_point("predict", len(new_prog.predict_ops)-1, feature_vars, state_vars)
            actual_last_op_out_type = last_op_spec.out_type
            if last_op_spec.is_elementwise and last_op_spec.out_type == "scalar":
                if any(vars_before_last.get(inp_n) == "vector" for inp_n in last_op.inputs): 
                     actual_last_op_out_type = "vector"

            if last_op.out != FINAL_PREDICTION_VECTOR_NAME or actual_last_op_out_type != "vector":
                if actual_last_op_out_type == "vector": 
                     # Rename the output of the current last op
                     new_prog.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, last_op.opcode, last_op.inputs)
                else: 
                    # The current last op doesn't produce a vector. Add an assign_vector.
                    # Vars available for this new assign_vector include the output of the current (non-vector) last op
                    vars_for_final_fix = {**vars_before_last, last_op.out: actual_last_op_out_type}
                    available_vectors = [vn for vn, vt in vars_for_final_fix.items() if vt == "vector"]
                     
                    if not available_vectors: # Fallback to feature vectors if no runtime vectors available
                        available_vectors = [fn for fn,ft in feature_vars.items() if ft == "vector"]
                        if not available_vectors and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES: # Further fallback
                             available_vectors = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES
                    
                    if available_vectors:
                        source_for_final_fix = rng.choice(available_vectors)
                        new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_final_fix,)))
                    else: # Absolute fallback: use a default feature if nothing else works
                        default_feat_vec_fallback = "opens_t" if "opens_t" in feature_vars else \
                                                  (CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "fallback_undefined_vector")
                        new_prog.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_feat_vec_fallback,)))

        elif not new_prog.predict_ops : # Predict block became empty
            default_feat_vec = next((vn for vn, vt in feature_vars.items() if vt == "vector"), None)
            if not default_feat_vec and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
                default_feat_vec = rng.choice(CROSS_SECTIONAL_FEATURE_VECTOR_NAMES)
            elif not default_feat_vec: # Absolute fallback
                default_feat_vec = "opens_t" 

            if default_feat_vec: 
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
            
            # After crossover, ensure the predict block ends correctly
            temp_feature_vars = AlphaProgram._get_default_feature_vars() 
            temp_state_vars = {} 
            
            # Check if predict_ops is empty *after* crossover but before adding final op
            if not child.predict_ops:
                default_vec_src = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t"
                child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_vec_src,)))
            else: # Predict_ops is not empty, check its last op
                vars_before_final_op_in_child = child.get_vars_at_point("predict", len(child.predict_ops)-1, temp_feature_vars, temp_state_vars)
                last_op_in_child = child.predict_ops[-1]
                last_op_spec_in_child = OP_REGISTRY[last_op_in_child.opcode]
                
                actual_last_op_out_type_child = last_op_spec_in_child.out_type
                if last_op_spec_in_child.is_elementwise and last_op_spec_in_child.out_type == "scalar":
                    if any(vars_before_final_op_in_child.get(inp_n) == "vector" for inp_n in last_op_in_child.inputs):
                        actual_last_op_out_type_child = "vector"

                if actual_last_op_out_type_child == "vector":
                    child.predict_ops[-1] = Op(FINAL_PREDICTION_VECTOR_NAME, last_op_in_child.opcode, last_op_in_child.inputs)
                else: # Last op doesn't produce vector, need to add assign_vector
                    vars_for_final_assign = {**vars_before_final_op_in_child, last_op_in_child.out: actual_last_op_out_type_child}
                    available_vectors = [vn for vn, vt in vars_for_final_assign.items() if vt == "vector"]
                    if not available_vectors:
                        available_vectors = [fn for fn,ft in temp_feature_vars.items() if ft == "vector"]
                        if not available_vectors and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES:
                            available_vectors = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES

                    source_for_assign = rng.choice(available_vectors) if available_vectors else (CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t")
                    child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_assign,)))

        elif other.predict_ops: # Child's predict_ops was empty, other's was not. Deepcopy other's predict_ops.
            child.predict_ops = copy.deepcopy(other.predict_ops) 
            # And ensure it's valid
            if child.predict_ops: # Should be true if other.predict_ops was true
                temp_feature_vars = AlphaProgram._get_default_feature_vars()
                temp_state_vars = {}
                vars_before_final_op_in_child = child.get_vars_at_point("predict", len(child.predict_ops)-1, temp_feature_vars, temp_state_vars)
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
                    if not available_vectors: available_vectors = [fn for fn,ft in temp_feature_vars.items() if ft == "vector"]
                    source_for_assign = rng.choice(available_vectors) if available_vectors else (CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t")
                    child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (source_for_assign,)))
            else: # This case implies other.predict_ops was also empty, or became empty after popping.
                  # Ensure child.predict_ops is not empty.
                default_vec_src = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t"
                child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_vec_src,)))
        
        # If predict_ops ended up empty for any reason, add a default final op
        if not child.predict_ops:
            default_vec_src = CROSS_SECTIONAL_FEATURE_VECTOR_NAMES[0] if CROSS_SECTIONAL_FEATURE_VECTOR_NAMES else "opens_t"
            child.predict_ops.append(Op(FINAL_PREDICTION_VECTOR_NAME, "assign_vector", (default_vec_src,)))


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
        
        if not self.predict_ops: # Should not happen due to fixes in random_program, mutate, crossover
            # print("Warning: predict_ops is empty during eval. This should be fixed by generation/mutation logic.")
            return np.full(n_stocks, np.nan) 
            
        for op_instance in self.predict_ops:
            try:
                op_instance.execute(buf, n_stocks)
            except Exception: 
                # This might indicate an issue with an op or invalid inputs.
                # For robustness, return NaNs, but ideally, this should be rare with type checking.
                return np.full(n_stocks, np.nan)


        if FINAL_PREDICTION_VECTOR_NAME not in buf:
            # This implies the predict block did not correctly produce the final named vector.
            # Should be caught by generation/mutation logic.
            # print(f"Warning: {FINAL_PREDICTION_VECTOR_NAME} not in buffer after predict_ops. Program: {self.to_string()}")
            return np.full(n_stocks, np.nan)

        s1_predictions_val = buf[FINAL_PREDICTION_VECTOR_NAME]

        if np.isscalar(s1_predictions_val): 
            s1_predictions_val = np.full(n_stocks, float(s1_predictions_val))
        
        if not isinstance(s1_predictions_val, np.ndarray) or s1_predictions_val.ndim != 1:
            # Output is not a 1D array as expected for a vector.
            # print(f"Warning: {FINAL_PREDICTION_VECTOR_NAME} is not a 1D ndarray. Type: {type(s1_predictions_val)}")
            return np.full(n_stocks, np.nan) 

        if s1_predictions_val.shape[0] != n_stocks:
            # Output vector length does not match n_stocks.
            # This could happen if a cross-sectional aggregator was misused or if resizing logic failed.
            # For now, returning NaN; might need more robust handling or ensuring ops always respect n_stocks for vector out.
            # print(f"Warning: {FINAL_PREDICTION_VECTOR_NAME} shape {s1_predictions_val.shape} does not match n_stocks {n_stocks}")
            # Let's try to resize/broadcast if it's a single value, otherwise NaN.
            if s1_predictions_val.size == 1:
                 s1_predictions_val = np.full(n_stocks, s1_predictions_val.item())
            else:
                 return np.full(n_stocks, np.nan)


        initial_state_keys = set(state.keys())
        vars_defined_in_update = set()

        for op_instance in self.update_ops:
            try:
                op_instance.execute(buf, n_stocks)
                vars_defined_in_update.add(op_instance.out)
            except Exception: 
                return np.full(n_stocks, np.nan) 

        # Persist relevant state variables
        for key in list(state.keys()): # Iterate over original keys to update
            if key in buf and key not in features_at_t: # Ensure it's a state var, not an input feature
                state[key] = buf[key]

        # Add newly defined state variables from update block
        for key in vars_defined_in_update:
            if key not in features_at_t: # Ensure it's not an input feature name
                 state[key] = buf[key]

        return np.nan_to_num(s1_predictions_val.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


    @property
    def size(self) -> int:
        return len(self.setup) + len(self.predict_ops) + len(self.update_ops)

    def to_string(self, max_len: int = 1000) -> str:
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