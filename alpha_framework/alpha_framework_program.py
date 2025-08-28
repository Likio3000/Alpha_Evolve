from __future__ import annotations
from dataclasses import dataclass, field
import copy
import hashlib
import json
import textwrap
from typing import Dict, List, Literal, Optional, Union
import numpy as np

# Imports from other framework modules
from .alpha_framework_types import (
    TypeId,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    FINAL_PREDICTION_VECTOR_NAME,
    OP_REGISTRY,  # Required for OpSpec access
)
from .alpha_framework_op import Op
from . import program_logic_generation
from .program_logic_generation import generate_random_program_logic
from .program_logic_variation import mutate_program_logic, crossover_program_logic


@dataclass
class AlphaProgram:
    """Container for a single alpha trading program.

    The program is split into three ordered lists of :class:`Op` instances:
    a setup block, a predict block and an update block.  The predict block is
    expected to end with an operation that produces the
    :data:`FINAL_PREDICTION_VECTOR_NAME` vector.
    """

    setup: List[Op] = field(default_factory=list)
    predict_ops: List[Op] = field(default_factory=list)
    update_ops: List[Op] = field(default_factory=list)

    _vars_info_cache: Optional[Dict[str, Dict[str, TypeId]]] = field(
        default=None, repr=False, compare=False
    )

    def _trace_vars_for_block(
        self, ops_block: List[Op], initial_vars: Dict[str, TypeId]
    ) -> Dict[str, TypeId]:
        """Infer resulting variable types after executing a block of operations.

        Parameters
        ----------
        ops_block : List[Op]
            Operations to simulate sequentially.
        initial_vars : Dict[str, TypeId]
            Known variable names and their types at the start of the block.

        Returns
        -------
        Dict[str, TypeId]
            Updated mapping of variable names to their inferred types.

        Notes
        -----
        Elementwise operations that nominally output scalars are promoted to
        return vectors when any input that the spec marks as scalar is provided
        a vector. This allows vector inputs to flow through elementwise ops
        without losing their vector nature.
        """

        current_vars = initial_vars.copy()
        for op_instance in ops_block:
            spec = OP_REGISTRY[op_instance.opcode]
            actual_out_type = spec.out_type
            if spec.is_elementwise and spec.out_type == "scalar":
                is_any_input_vector = False
                for i, in_name in enumerate(op_instance.inputs):
                    input_var_type = current_vars.get(in_name)
                    if input_var_type == "vector":
                        # Check if the op spec expects a scalar for this vector input
                        if spec.in_types[i] == "scalar":
                            is_any_input_vector = True
                            break
                if is_any_input_vector:
                    actual_out_type = "vector"
            current_vars[op_instance.out] = actual_out_type
        return current_vars

    def get_vars_at_point(
        self,
        block_name: Literal["setup", "predict", "update"],
        op_index: int,
        feature_vars: Dict[str, TypeId],
        state_vars: Dict[str, TypeId],
    ) -> Dict[str, TypeId]:
        """Return variables available before a given operation.

        Parameters
        ----------
        block_name:
            Which block to inspect (``"setup"``, ``"predict"`` or ``"update"``).
        op_index:
            Index of the operation within that block.
        feature_vars:
            Mapping of feature variable names to types.
        state_vars:
            Mapping of state variable names to types.

        Returns
        -------
        Dict[str, TypeId]
            Combined dictionary of the base variables and those defined up to the
            specified point.
        """

        available_vars = {}
        base_vars = {**feature_vars, **state_vars}

        if block_name == "setup":
            available_vars = self._trace_vars_for_block(
                self.setup[:op_index], base_vars
            )
        elif block_name == "predict":
            vars_after_setup = self._trace_vars_for_block(self.setup, base_vars)
            available_vars = self._trace_vars_for_block(
                self.predict_ops[:op_index], vars_after_setup
            )
        elif block_name == "update":
            vars_after_setup = self._trace_vars_for_block(self.setup, base_vars)
            vars_after_predict = self._trace_vars_for_block(
                self.predict_ops, vars_after_setup
            )
            merged_context_before_update = {**vars_after_predict}
            available_vars = self._trace_vars_for_block(
                self.update_ops[:op_index], merged_context_before_update
            )

        return {
            **base_vars,
            **available_vars,
        }  # Return merged available variables with base

    @classmethod
    def random_program(
        cls,
        feature_vars: Dict[str, TypeId],
        state_vars: Dict[str, TypeId],
        max_total_ops: int = 32,
        rng: Optional[np.random.Generator] = None,
        max_setup_ops: int = program_logic_generation.MAX_SETUP_OPS,
        max_predict_ops: int = program_logic_generation.MAX_PREDICT_OPS,
        max_update_ops: int = program_logic_generation.MAX_UPDATE_OPS,
        ops_split_jitter: float = 0.0,
    ) -> "AlphaProgram":
        """Create a random, type-correct program.

        Parameters
        ----------
        feature_vars, state_vars:
            Dictionaries mapping variable names to their types.
        max_total_ops:
            Maximum number of operations across all blocks.
        rng:
            Optional random number generator used for reproducibility.
        max_setup_ops, max_predict_ops, max_update_ops:
            Per-block limits overriding the defaults from the paper.

        Returns
        -------
        AlphaProgram
            Newly generated program instance.
        """

        return generate_random_program_logic(
            cls,
            feature_vars,
            state_vars,
            max_total_ops,
            rng,
            max_setup_ops,
            max_predict_ops,
            max_update_ops,
            ops_split_jitter,
        )

    def copy(self) -> "AlphaProgram":
        """Return a deep copy of this program."""

        new_prog = AlphaProgram(
            setup=copy.deepcopy(self.setup),
            predict_ops=copy.deepcopy(self.predict_ops),
            update_ops=copy.deepcopy(self.update_ops),
        )
        # _vars_info_cache is not copied as it's a cache for the specific instance
        return new_prog

    def mutate(
        self,
        feature_vars: Dict[str, TypeId],
        state_vars: Dict[str, TypeId],
        prob_add: float = 0.2,
        prob_remove: float = 0.2,
        prob_change_op: float = 0.3,
        prob_change_inputs: float = 0.3,
        max_total_ops: int = 87,
        rng: Optional[np.random.Generator] = None,
        max_setup_ops: int = program_logic_generation.MAX_SETUP_OPS,
        max_predict_ops: int = program_logic_generation.MAX_PREDICT_OPS,
        max_update_ops: int = program_logic_generation.MAX_UPDATE_OPS,
    ) -> "AlphaProgram":
        """Return a mutated copy of this program."""

        return mutate_program_logic(
            self,
            feature_vars,
            state_vars,
            prob_add,
            prob_remove,
            prob_change_op,
            prob_change_inputs,
            max_total_ops,
            rng,
            max_setup_ops,
            max_predict_ops,
            max_update_ops,
        )

    def crossover(
        self,
        other: "AlphaProgram",
        rng: Optional[np.random.Generator] = None,
        max_setup_ops: int = program_logic_generation.MAX_SETUP_OPS,
        max_predict_ops: int = program_logic_generation.MAX_PREDICT_OPS,
        max_update_ops: int = program_logic_generation.MAX_UPDATE_OPS,
    ) -> "AlphaProgram":
        """Create offspring from this program and ``other``."""

        return crossover_program_logic(
            self,
            other,
            rng,
            max_setup_ops,
            max_predict_ops,
            max_update_ops,
        )

    @staticmethod
    def _get_default_feature_vars() -> Dict[str, TypeId]:
        """Return a mapping of default feature names to ``"vector"`` type."""

        default_vars = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
        # Do not include constant scalar features by default.
        return default_vars

    def new_state(self) -> Dict[str, Union[np.ndarray, float]]:
        """Return the initial program state.

        Override in subclasses if more complex state is required.
        """

        return {}

    def eval(
        self,
        features_at_t: Dict[str, Union[np.ndarray, float]],
        state: Dict[str, Union[np.ndarray, float]],
        n_stocks: int,
    ) -> np.ndarray:
        """Execute the program for one timestep.

        Parameters
        ----------
        features_at_t:
            Current feature values keyed by variable name.
        state:
            Mutable state dictionary that will be updated in-place.
        n_stocks:
            Number of stocks for vector operations.

        Returns
        -------
        np.ndarray
            Prediction vector of length ``n_stocks``. Invalid executions result
            in a vector of ``NaN`` values.
        """

        self._vars_info_cache = None  # Clear cache at start of eval

        buf: Dict[str, Union[np.ndarray, float]] = {**features_at_t, **state}

        for op_instance in self.setup:
            op_instance.execute(buf, n_stocks)

        if not self.predict_ops:
            # Fallback if predict_ops is empty, though generation/mutation should prevent this.
            # print("Warning: predict_ops is empty during eval.") # Optional: for debugging
            return np.full(n_stocks, np.nan)

        for op_instance in self.predict_ops:
            try:
                op_instance.execute(buf, n_stocks)
            except Exception:  # Broad exception for robustness during eval
                # print(f"Error during predict_op execution: {op_instance} with error {e}") # Optional
                return np.full(n_stocks, np.nan)

        if FINAL_PREDICTION_VECTOR_NAME not in buf:
            # print(f"Warning: {FINAL_PREDICTION_VECTOR_NAME} not in buffer. Program: {self.to_string()}") # Optional
            return np.full(n_stocks, np.nan)

        s1_predictions_val = buf[FINAL_PREDICTION_VECTOR_NAME]

        # Ensure output is a correctly shaped 1D numpy array
        if np.isscalar(s1_predictions_val):
            s1_predictions_val = np.full(n_stocks, float(s1_predictions_val))

        if (
            not isinstance(s1_predictions_val, np.ndarray)
            or s1_predictions_val.ndim != 1
        ):
            # print(f"Warning: {FINAL_PREDICTION_VECTOR_NAME} is not a 1D ndarray. Type: {type(s1_predictions_val)}") # Optional
            return np.full(n_stocks, np.nan)

        if s1_predictions_val.shape[0] != n_stocks:
            # print(f"Warning: {FINAL_PREDICTION_VECTOR_NAME} shape {s1_predictions_val.shape} != n_stocks {n_stocks}") # Optional
            if (
                s1_predictions_val.size == 1
            ):  # Attempt to broadcast if it's a scalar-like array
                s1_predictions_val = np.full(n_stocks, s1_predictions_val.item())
            else:  # Otherwise, it's a mismatch
                return np.full(n_stocks, np.nan)

        # State update logic
        # initial_state_keys = set(state.keys()) # Not strictly needed with current buf init
        vars_defined_in_update = set()

        for op_instance in self.update_ops:
            try:
                op_instance.execute(buf, n_stocks)
                vars_defined_in_update.add(op_instance.out)
            except Exception:  # Broad exception
                # print(f"Error during update_op execution: {op_instance} with error {e}") # Optional
                # Don't necessarily return NaN here; prediction might be valid, state update fails for this step.
                pass

        # Persist relevant state variables from buf back to state
        # Update existing state keys that were modified by any block
        for key in list(state.keys()):  # Iterate over original state keys
            if (
                key in buf and key not in features_at_t
            ):  # Must be a state var, not an input feature
                state[key] = buf[key]

        # Add newly defined state variables from update block (if not already features or existing state)
        for key in vars_defined_in_update:
            if key not in features_at_t:  # Ensure it's not an input feature name
                state[key] = buf[
                    key
                ]  # This will add new state or update existing if redefined in update

        return np.nan_to_num(
            s1_predictions_val.astype(float), nan=0.0, posinf=0.0, neginf=0.0
        )

    @property
    def size(self) -> int:
        """Total number of operations across all blocks."""

        return len(self.setup) + len(self.predict_ops) + len(self.update_ops)

    def to_string(self, max_len: int = 1000) -> str:
        """Return a short text representation of the program."""

        txt_parts = []
        if self.setup:
            txt_parts.append(f"S[{';'.join(map(str, self.setup))}]")
        if self.predict_ops:
            txt_parts.append(f"P[{';'.join(map(str, self.predict_ops))}]")
        if self.update_ops:
            txt_parts.append(f"U[{';'.join(map(str, self.update_ops))}]")

        full_txt = " >> ".join(txt_parts)
        return textwrap.shorten(full_txt, width=max_len, placeholder="...")

    @property
    def fingerprint(self) -> str:
        """Stable hash representing the program structure."""

        serial = {
            "setup": [(o.out, o.opcode, o.inputs) for o in self.setup],
            "predict": [(o.out, o.opcode, o.inputs) for o in self.predict_ops],
            "update": [(o.out, o.opcode, o.inputs) for o in self.update_ops],
        }
        return hashlib.sha1(
            json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def prune(self) -> "AlphaProgram":
        """Remove operations whose outputs are never used."""

        # Walk blocks backwards to track which variables are required
        # for producing the final prediction vector.  Any operation that does
        # not contribute to that set is discarded.
        useful = {FINAL_PREDICTION_VECTOR_NAME}

        def _prune_block(block):
            kept_rev = []
            for op in reversed(block):
                if op.out in useful:
                    useful.update(op.inputs)
                    kept_rev.append(op)
            kept_rev.reverse()
            return kept_rev

        self.predict_ops = _prune_block(self.predict_ops)
        self.setup = _prune_block(self.setup)
        self.update_ops = _prune_block(self.update_ops)

        self._vars_info_cache = None
        return self
