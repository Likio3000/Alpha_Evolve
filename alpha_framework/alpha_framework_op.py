from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Union # Added Union
import numpy as np

# Imports from other framework modules
from .alpha_framework_types import OP_REGISTRY # OpSpec is implicitly used via OP_REGISTRY
from .utils import ensure_vector_shape, broadcast_to_vector

# ---------------------------------------------------------------------------
# 3 – Instruction container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Op:
    out: str
    opcode: str
    inputs: Tuple[str, ...]

    """Single operation used by :class:`AlphaProgram`."""

    def execute(self, buf: Dict[str, Union[np.ndarray, float]], n_stocks: int):
        """Execute this operation, updating ``buf`` in-place."""

        spec = OP_REGISTRY[self.opcode]

        processed_args = []
        for i, in_name in enumerate(self.inputs):
            arg_val = buf.get(in_name)
            if arg_val is None:
                raise KeyError(f"Op '{self.opcode}' input variable '{in_name}' not found in buffer. Buffer keys: {list(buf.keys())}")

            expected_type = spec.in_types[i]

            if expected_type == "scalar":
                if isinstance(arg_val, np.ndarray):
                    if arg_val.size == 1:
                        processed_args.append(arg_val.item())
                    elif spec.is_elementwise and arg_val.ndim == 1:
                        # Allow passing vector into a scalar slot for elementwise ops
                        processed_args.append(arg_val)
                    else:
                        processed_args.append(np.mean(arg_val) if arg_val.size > 0 else 0.0)
                elif np.isscalar(arg_val):
                    processed_args.append(float(arg_val))
                else:
                    raise TypeError(
                        f"Op '{self.opcode}' input '{in_name}' expected scalar, got {type(arg_val)} value {arg_val}"
                    )

            elif expected_type == "vector":
                if np.isscalar(arg_val):
                    processed_args.append(broadcast_to_vector(arg_val, n_stocks))
                elif isinstance(arg_val, np.ndarray) and arg_val.ndim == 1:
                    processed_args.append(
                        ensure_vector_shape(
                            arg_val, n_stocks, allow_mismatch_for_aggregator=spec.is_cross_sectional_aggregator
                        )
                    )
                else:
                    raise TypeError(
                        f"Op '{self.opcode}' input '{in_name}' expected vector, got {type(arg_val)} value {arg_val}"
                    )
            elif expected_type == "matrix":
                if not (isinstance(arg_val, np.ndarray) and arg_val.ndim == 2):
                    raise TypeError(f"Op '{self.opcode}' input '{in_name}' expected matrix, got {type(arg_val)} value {arg_val}")
                processed_args.append(arg_val)

        result = spec.func(*processed_args)

        if spec.is_elementwise and spec.out_type == "scalar" and isinstance(result, np.ndarray) and result.ndim == 1:
            pass
        elif spec.out_type == "vector":
            result = ensure_vector_shape(
                result,
                n_stocks,
                allow_mismatch_for_aggregator=spec.is_cross_sectional_aggregator,
            )

        buf[self.out] = result

    def __str__(self):
        return f"{self.out} = {self.opcode}({', '.join(self.inputs)})"
