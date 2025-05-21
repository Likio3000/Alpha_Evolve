from __future__ import annotations

"""alpha_program.py
========================================
Fully‑featured *instruction‑list* representation of an alpha program that
matches the usage expected by **evolve_alphas.py**:

* `AlphaProgram.random_program()` – class‑method seed generator
* `mutate()` / `crossover()` / `copy()` – evolutionary operators
* Cheap `size`, `depth` (≈ longest dependency chain), `to_string()` helpers
* Robust `fingerprint` for duplicate filtering cache

The numerical operator registry remains **NumPy‑only** and intentionally
minimal; extend it as you add RelationOps, rolling windows, etc.
"""

from dataclasses import dataclass, field, replace
import copy
import hashlib
import json
import random
from typing import Callable, Dict, List, Tuple, Literal, Sequence, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1 ‑‑ Type system (very light‑weight) --------------------------------------
# ---------------------------------------------------------------------------

TypeId = Literal["scalar", "vector", "matrix"]


@dataclass
class OpSpec:
    func: Callable
    in_types: Tuple[TypeId, ...]
    out_type: TypeId


# Global registry – populated at import time.
OP_REGISTRY: Dict[str, OpSpec] = {}


def register_op(name: str, *, in_types: Tuple[TypeId, ...], out: TypeId):
    """Decorator to register a NumPy‑based operator in *OP_REGISTRY*."""

    def _wrapper(fn: Callable):
        if name in OP_REGISTRY:
            raise KeyError(f"opcode '{name}' registered twice")
        OP_REGISTRY[name] = OpSpec(fn, in_types, out)
        return fn

    return _wrapper


# ---------------------------------------------------------------------------
# 2 ‑‑ Primitive operators ---------------------------------------------------
# ---------------------------------------------------------------------------
# Scalar ↔ scalar ------------------------------------------------------------


@register_op("add", in_types=("scalar", "scalar"), out="scalar")
def _add(a, b):
    return a + b


@register_op("sub", in_types=("scalar", "scalar"), out="scalar")
def _sub(a, b):
    return a - b


@register_op("mul", in_types=("scalar", "scalar"), out="scalar")
def _mul(a, b):
    return a * b


@register_op("div", in_types=("scalar", "scalar"), out="scalar")
def _div(a, b):
    return a / (b if np.abs(b) > 1e-12 else 1e-12)


@register_op("tanh", in_types=("scalar",), out="scalar")
def _tanh(a):
    return np.tanh(a)


@register_op("sign", in_types=("scalar",), out="scalar")
def _sign(a):
    return np.sign(a)


@register_op("neg", in_types=("scalar",), out="scalar")
def _neg(a):
    return -a


@register_op("abs", in_types=("scalar",), out="scalar")
def _abs(a):
    return np.abs(a)


@register_op("identity", in_types=("scalar",), out="scalar")
def _identity(x):
    return x


# Vector ops ----------------------------------------------------------------


@register_op("mean", in_types=("vector",), out="scalar")
def _vmean(v):
    return float(np.mean(v))


@register_op("std", in_types=("vector",), out="scalar")
def _vstd(v):
    return float(np.std(v, ddof=0))


@register_op("rank", in_types=("vector",), out="vector")
def _vrank(v):
    r = np.argsort(np.argsort(v)) / (len(v) - 1 + 1e-12)
    return r * 2.0 - 1.0


@register_op("relation_demean", in_types=("scalar", "vector"), out="scalar")
def _relation_demean(x, peer_vec):
    return x - float(np.mean(peer_vec))


# Matrix ops ----------------------------------------------------------------


@register_op("matmul", in_types=("matrix", "vector"), out="vector")
def _matmul(m, v):
    return m @ v


@register_op("transpose", in_types=("matrix",), out="matrix")
def _transpose(m):
    return m.T


# Extraction ops -------------------------------------------------------------


@register_op("get_row", in_types=("matrix", "scalar"), out="vector")
def _get_row(m, idx):
    return m[int(idx)]


@register_op("get_col", in_types=("matrix", "scalar"), out="vector")
def _get_col(m, idx):
    return m[:, int(idx)]


# ---------------------------------------------------------------------------
# 3 ‑‑ Instruction & program container --------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class Op:
    out: str
    opcode: str
    inputs: Tuple[str, ...]

    def execute(self, buf: Dict[str, np.ndarray]):
        spec = OP_REGISTRY[self.opcode]
        args = [buf[name] for name in self.inputs]
        buf[self.out] = spec.func(*args)

    # helper string
    def __str__(self):
        return f"{self.out} = {self.opcode}({', '.join(self.inputs)})"


@dataclass
class AlphaProgram:
    setup: List[Op] = field(default_factory=list)
    predict_ops: List[Op] = field(default_factory=list)
    update_ops: List[Op] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # Class‑level helpers
    # ---------------------------------------------------------------------

    @classmethod
    def random_program(cls, max_ops: int = 32, rng: Optional[np.random.Generator] = None) -> "AlphaProgram":
        rng = rng or np.random.default_rng()
        prog = cls()
        # simplistic: random scalar chain feeding from OHLC terminals
        SCALAR_TERMINALS = ["open", "high", "low", "close", "range"]
        cur = rng.choice(SCALAR_TERMINALS)
        tmp_idx = 0

        def new_tmp():
            nonlocal tmp_idx
            tmp_idx += 1
            return f"t{tmp_idx}"

        n_ops = rng.integers(4, max_ops)
        for _ in range(n_ops):
            if rng.random() < 0.3:  # unary
                opc = rng.choice(["tanh", "sign", "neg", "abs"])
                out = new_tmp()
                prog.predict_ops.append(Op(out, opc, (cur,)))
                cur = out
            else:  # binary
                opc = rng.choice(["add", "sub", "mul", "div"])
                rhs = rng.choice(SCALAR_TERMINALS)
                out = new_tmp()
                prog.predict_ops.append(Op(out, opc, (cur, rhs)))
                cur = out
        # final mapping → s1
        prog.predict_ops.append(Op("s1", "identity", (cur,)))
        return prog

    # ---------------------------------------------------------------------
    # Evolution helpers
    # ---------------------------------------------------------------------

    def copy(self) -> "AlphaProgram":
        return copy.deepcopy(self)

    def mutate(self, prob: float = 0.1, max_ops: int = 32) -> "AlphaProgram":
        rng = np.random.default_rng()
        new_prog = self.copy()
        if rng.random() < prob and len(new_prog.predict_ops) < max_ops:
            # insert a random unary op at random position
            idx = rng.integers(0, len(new_prog.predict_ops))
            cur_tmp = f"m{rng.integers(1e6)}"
            src = rng.choice([op.out for op in new_prog.predict_ops])
            opc = rng.choice(["tanh", "sign", "neg", "abs"])
            new_prog.predict_ops.insert(idx, Op(cur_tmp, opc, (src,)))
            # reconnect downstream ops that used src with 50% chance
            for op in new_prog.predict_ops[idx + 1 :]:
                if src in op.inputs and rng.random() < 0.5:
                    op.inputs = tuple(cur_tmp if i == src else i for i in op.inputs)
        return new_prog

    def crossover(self, other: "AlphaProgram") -> "AlphaProgram":
        rng = np.random.default_rng()
        child = self.copy()
        if not other.predict_ops:
            return child
        # swap a contiguous slice of predict_ops
        a, b = sorted(rng.choice(len(child.predict_ops), 2, replace=False))
        slice_other = copy.deepcopy(other.predict_ops[a : b + 1])
        child.predict_ops[a : b + 1] = slice_other
        return child

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------

    def new_state(self) -> Dict[str, np.ndarray]:
        return {}

    def eval(self, features: Dict[str, np.ndarray], state: Dict[str, np.ndarray]):
        """Run one full step (setup → predict → update) and return *s1*."""
        buf: Dict[str, np.ndarray] = {**features, **state}
        # ‑ setup once per call (cheap because list usually empty)
        for op in self.setup:
            op.execute(buf)
        # ‑ predict
        for op in self.predict_ops:
            op.execute(buf)
        s1 = float(buf["s1"])
        # ‑ update (stateful)
        for op in self.update_ops:
            op.execute(buf)
        # write back non‑feature keys into *state*
        for k, v in buf.items():
            if k not in features:
                state[k] = v
        return s1

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.setup) + len(self.predict_ops) + len(self.update_ops)

    def to_string(self, max_len: int = 120) -> str:
        txt = "; ".join(str(op) for op in self.predict_ops)
        return txt if len(txt) <= max_len else txt[: max_len - 3] + "…"

    # fingerprint
    @property
    def fingerprint(self) -> str:
        serial = [(o.out, o.opcode, o.inputs) for o in self.predict_ops]
        return hashlib.sha1(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
